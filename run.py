import os
import sys
import cv2
import json
import ffmpeg
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime
from tabulate import tabulate
from multiprocessing import freeze_support
os.environ['CUDA_MODULE_LOADING'] = 'LAZY' 

import daram

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval

# Setting up loguru logger to save all logs, including info, warning, and error, to a file in the 'runs' folder
def setup_logger():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = os.path.join("runs", current_time)
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, "result.log")
    logger.add(log_path, format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip")
    return log_folder

def parse_args():
    parser = argparse.ArgumentParser("tracker validation inference parser")
    # dataset options
    parser.add_argument("--dataset", type=str, default="handsome-kith_seongsu", choices=["MOT17", "MOT20", "DanceTrack", "amorepacific_seongsu", "handsome-kith_seongsu, shinsegae_gangnam-b1f"])
    parser.add_argument("--tag", type=str, default="every")
    parser.add_argument("--video_root", type=str, default="/videoset", help="root path of video")

    # tracker options
    parser.add_argument("--tracker", type=str, default="visiou", choices=["vispose", "visiou"])
    parser.add_argument("--reid_model", type=str, default="new", choices=["new", "old"])

    # evaluation options
    parser.add_argument('--eval', default=True, action='store_true', help='Run evaluation')
    parser.add_argument('--GT_FOLDER', default="/user/gt", help='Location of GT data')
    return parser.parse_args()


def sort_key(x):
    x_list = x.split(",")
    return (int(x_list[0]), int(x_list[1]))


def daram_to_mot(daram_data, save_mot_path):
    mot_results = []
    for fid, v in daram_data.items():
        bbox_dict = v['person_joints']
        for tid, res in bbox_dict.items():
            bbox = res['bbox']
            width = bbox[1][0] - bbox[0][0]
            height = bbox[1][1] - bbox[0][1]
            mot_results.append(
                f"{int(fid)+1},{tid},{bbox[0][0]:.2f},{bbox[0][1]:.2f},{width:.2f},{height:.2f},-1,-1,-1\n"
            )       
    mot_results.sort(key=sort_key)  
    
    with open(save_mot_path, 'w') as f:
        f.writelines(mot_results)


def make_video(input_frame_path, output_video_path, frame_width=1920, frame_height=1080):
    video_info = input_frame_path[:-10] + "seqinfo.ini"
    with open(video_info, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        framerate = int(lines[3].split('=')[-1])

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    if not os.path.exists(output_video_path):
        (
            ffmpeg
            .input(input_frame_path, pattern_type='glob', framerate=framerate)
            .filter('scale', frame_width, frame_height)
            .output(output_video_path, crf=17, pix_fmt='yuv420p')
            .overwrite_output()
            .run()
        )
        print(f"Video saved at: {output_video_path}")


def trim_video(input_video_path, output_video_path, duration_seconds):
    (
        ffmpeg
        .input(input_video_path)
        .output(output_video_path, t=duration_seconds)
        .overwrite_output()
        .run()
    )
    print(output_video_path)


def extract_frames(video_path, output_dir, fps_target=10, max_frames=1500):
    os.makedirs(output_dir, exist_ok=True)

    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    fps = eval(video_info['r_frame_rate'])  # 비디오의 실제 FPS

    save_interval = max(1, round(fps / fps_target))

    process = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    frame_size = width * height * 3
    saved_count = 0
    for i in range(num_frames):
        if i >= max_frames:
            break
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break

        if i % save_interval == 0:
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(output_dir, f'{saved_count:06d}.jpg')
            cv2.imwrite(frame_path, frame_bgr)
            saved_count += 1

        if i % 100 == 0:  # 진행 상황 출력
            print(f"Processed {i}/{num_frames} frames, Saved {saved_count} frames")

    process.stdout.close()
    process.wait()
    print(f"Extracted {saved_count} frames at approximately {fps_target} FPS")


def evaluation(args, save_tracker_result_path):
    tracker_folder = "/".join(save_tracker_result_path.split("/")[:2])
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    config['GT_FOLDER'] = args.GT_FOLDER
    config['TRACKERS_FOLDER'] = tracker_folder
    config['BENCHMARK'] = args.dataset
    config['SPLIT_TO_EVAL'] = args.tag

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    print(f"Finished Evaluating Benchmark: {args.dataset}, Split: {args.tag}\n")

    # 각 metric 별로 평균값을 구해보자
    metric_values = {}
    for fname, res in output_res['MotChallenge2DBox']['last_benchmark'].items():
        HOTA = res['pedestrian']['HOTA']['HOTA'].mean()
        MOTA = res['pedestrian']['CLEAR']['MOTA']
        IDSW = res['pedestrian']['CLEAR']['IDSW']
        IDF1 = res['pedestrian']['Identity']['IDF1']
        metric_values[fname] = [HOTA, MOTA, IDSW, IDF1]
    mean_values = np.mean(list(metric_values.values()), axis=0)

    return mean_values


def plot_line_graph(results, output_dir, x_param='appearance_thresh', y_param='HOTA', group_by='MAX_CDIST_TH'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Group the data by the `group_by` parameter
    for group_val, group_data in results.groupby(group_by):
        plt.plot(
            group_data[x_param],
            group_data[y_param],
            marker='o',
            label=f'{group_by}={group_val}'
        )

    # Add labels, legend, and title
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} vs. {x_param}')
    plt.legend(title=group_by)
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(output_dir, f'{y_param}_line_graph.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Line graph saved to: {output_path}")


def main(args):
    log_folder = setup_logger()

    if args.dataset not in ["MOT17", "MOT20", "DanceTrack", "PersonPath22", "VOD", "SUP1", "SUP2"]:
        private = True
    else:
        private = False

    if private:
        if args.dataset == "amorepacific_seongsu":
            print("amorepacific")
            video_sets = [
                # {'timestamp': '1719034201464', 'location': 'amorepacific_seoungsu', 'date': '20240622', 'cam': '6'}
                {'timestamp': '1719045002590', 'location': 'amorepacific_seoungsu', 'date': '20240622', 'cam': '10'}
                ]
        elif args.dataset == "handsome-kith_seongsu":
            print("handsome-kith")
            video_sets = [{'timestamp': '1720249202989', 'location': 'handsome-kith_seongsu', 'date': '20240706', 'cam': '211'}]

        elif args.dataset == "shinsegae_gangnam-b1f":
            print("shinsegae")
            video_sets = [{'timestamp': '1691908205788', 'location': 'shinsegae_gangnam-b1f', 'date': '20230813', 'cam': '3'}]

        output_video_paths = []
        for video_set in video_sets:
            base_path = f"{args.video_root}/{video_set['location']}/{video_set['date']}/{video_set['cam']}/{video_set['timestamp']}.mp4"
            output_video_path = f"/tmp/{video_set['cam']}_{video_set['timestamp']}.mp4"
            output_video_paths.append(output_video_path)
            if not os.path.exists(output_video_path):
                trim_video(base_path, output_video_path, 150)
            tmp_folder = f"/tmp/{video_set['cam']}_{video_set['timestamp']}"
            frame_dir = os.path.join(tmp_folder, "frames")
            if not os.path.exists(frame_dir):
                extract_frames(f"/tmp/{video_set['cam']}_{video_set['timestamp']}.mp4", frame_dir, fps_target=10, max_frames=1500)
    else:
        if "VOD" in args.dataset:
            print("vod")
            base_path = f"/maydrive/tracker_benchmarks/{args.dataset}/DataSet"
            project_folder = [p for p in os.listdir(base_path)]
            print(project_folder)

        elif "MOT" in args.dataset:
            base_path = f"/maydrive/tracker_benchmarks/{args.dataset}/{args.tag}"
            if args.dataset == "MOT17":
                project_folder = [p for p in os.listdir(base_path) if p.endswith("-SDP")]
            else:
                project_folder = [p for p in os.listdir(base_path)]

            project_folder = [p for p in project_folder if not os.path.exists(f"results/tracker_{args.tracker}/{args.dataset}-{args.tag}/last_benchmark/data/{p}.txt")]
            print(project_folder)

            output_video_paths = []
            for f in project_folder:
                frame_dir = os.path.join(os.path.join(base_path, f, "img1"), '*.jpg')
                tmp_folder = f"/tmp/{f}"
                output_video_path = f"/tmp/{args.dataset}/{f}.mp4"
                make_video(frame_dir, output_video_path)
                output_video_paths.append(output_video_path)

        elif "DanceTrack" in args.dataset:
            base_path = f"/maydrive/tracker_benchmarks/{args.dataset}/{args.tag}"
            project_folder = [p for p in os.listdir(base_path)]

            # f"results/tracker_{args.tracker}/{args.dataset}-{args.tag}/last_benchmark/data" 여기 없는 파일만 돌리기
            project_folder = [p for p in project_folder if not os.path.exists(f"results/tracker_{args.tracker}/{args.dataset}-{args.tag}/last_benchmark/data/{p}.txt")]
            print(project_folder)

            output_video_paths = []
            for f in project_folder:
                frame_dir = os.path.join(os.path.join(base_path, f, "img1"), '*.jpg')
                tmp_folder = f"/tmp/{f}"
                output_video_path = f"/tmp/{args.dataset}/{f}.mp4"
                make_video(frame_dir, output_video_path)
                output_video_paths.append(output_video_path)
                output_video_paths.sort()

    ## DETECTOR config
    detect_config = daram.core.detector.base_config.FullBodyDetectorConfig()
    detect_config.model_path = "weights/yolox_x_pretrained.engine"
    detect_config.inference_fps = 10

    ## TRACKER config
    if args.tracker == "vispose":
        tracker_config = daram.core.tracker.VisPoseTrackingModuleConfig()
    else:
        tracker_config = daram.core.tracker.VisIOUTrackingModuleConfig()
    tracker_config.param_preset = "yolox"
    tracker_config.inference_fps = 5
    tracker_config.BYTE_SCORE_HIGH = 0.1
    tracker_config.BATCH_SIZE = 8
    tracker_config.cluster_interval = 600
    # tracker_config.is_use_cluster = False

    if args.reid_model == "new":
        tracker_config.feature_extractor_type = "vit-small-ics"
        tracker_config.model_path = "weights/vit-small-ics_v2.engine"
        # tracker_config.appearance_thresh = 0.20
        # tracker_config.MAX_CDIST_TH = 0.20
    elif args.reid_model == "old":
        tracker_config.model_path = "weights/1635581407_43eph.engine" ###for 3.10.5
        # tracker_config.appearance_thresh = 0.25
        # tracker_config.MAX_CDIST_TH = 0.155
    else:
        raise ValueError("Invalid reid model")
    
    best_hota = 0
    all_results = []

    for appearance_thresh in np.arange(0.05, 0.25, 0.05):
        tracker_config.appearance_thresh = round(appearance_thresh, 2)
        for MAX_CDIST_TH in np.arange(0.05, 0.25, 0.05):
            tracker_config.MAX_CDIST_TH = round(MAX_CDIST_TH, 2)

            logger.info(f"\n\nappearance_thresh: {appearance_thresh}, MAX_CDIST_TH: {MAX_CDIST_TH}")

            # Run detector and tracker
            for output_video_path in output_video_paths:
                print("####################################### {} START #######################################".format(output_video_path))
                video = daram.Video(output_video_path)
                json_save_path = os.path.join(tmp_folder, "results")
                os.makedirs(json_save_path, exist_ok=True)
                video.data.set_path(json_save_path)

                detector = daram.core.detector.BodyPoseDetector(
                    video, detect_config
                    )
                detector.run()
                detector.save(os.path.join(video.data.path, "detection_result_yolox.json"))
                video.data.add_json(
                    "detection_result", os.path.join(video.data.path, "detection_result_yolox.json")
                )

                if args.tracker == "vispose":
                    tracker = daram.core.tracker.VisPoseTrackingModule(video, tracker_config)
                elif args.tracker == "visiou":
                    tracker = daram.core.tracker.VisIOUTrackingModule(video, tracker_config)
                else:
                    raise ValueError("Invalid tracker name")

                try:
                    tracker.run()
                    tracker.save_by_frame(os.path.join(json_save_path, f"tracking_result_by_frame_{args.dataset}.json"))
                    tracker.save_by_object(
                        os.path.join(video.data.path, f"tracking_result_by_object_{args.dataset}.json")
                    )

                    with open(f"{json_save_path}/tracking_result_by_frame_{args.dataset}.json") as f:
                        daram_data = json.load(f)
                    
                    save_tracker_result_path = f"results/tracker_{args.tracker}/{args.dataset}-{args.tag}/last_benchmark/data"
                    os.makedirs(save_tracker_result_path, exist_ok=True)
                    mot_save_path = output_video_path.split("/")[-1][:-4] + ".txt"
                    daram_to_mot(daram_data, os.path.join(save_tracker_result_path, mot_save_path))
                    print("Saved mot file")
                except Exception as e:
                    print(e)

            if os.path.exists(os.path.join(args.GT_FOLDER, f"{args.dataset}-{args.tag}")):
                mean_values = evaluation(args, save_tracker_result_path)

                # Store result
                all_results.append({
                    "appearance_thresh": appearance_thresh,
                    "MAX_CDIST_TH": MAX_CDIST_TH,
                    "HOTA": mean_values[0],
                    "MOTA": mean_values[1],
                    "IDSW": mean_values[2],
                    "IDF1": mean_values[3]
                })

                if mean_values[0] > best_hota:
                    best_hota = mean_values[0]
                    best_appearance_thresh = appearance_thresh
                    best_MAX_CDIST_TH = MAX_CDIST_TH
                    print(f"Best HOTA: {best_hota}, Best appearance_thresh: {best_appearance_thresh}, Best MAX_CDIST_TH: {best_MAX_CDIST_TH}")

    # Create a DataFrame from the results
    df_results = pd.DataFrame(all_results)

    # Sort by HOTA to get the best configuration at the top
    # df_results = df_results.sort_values(by="HOTA", ascending=False)

    # Display the table using tabulate for better formatting
    logger.info("\n####################################### Summary #######################################")
    logger.info(tabulate(df_results, headers='keys', tablefmt='pretty', floatfmt=".2f"))

    # Display the best configuration
    logger.info(f"Best HOTA: {best_hota}")
    logger.info(f"Best appearance_thresh: {best_appearance_thresh}")
    logger.info(f"Best MAX_CDIST_TH: {best_MAX_CDIST_TH}")

    plot_line_graph(df_results, log_folder, x_param='appearance_thresh', y_param='HOTA', group_by='MAX_CDIST_TH')

if __name__ == "__main__":
    freeze_support()
    args = parse_args()
    main(args)