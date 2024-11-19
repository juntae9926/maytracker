import json
import os
import argparse
import cv2
import numpy as np

import ffmpeg
import tempfile

import daram

MAYDRIVE_PATH = '/work/videoset'

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



def main(args):

    if args.dataset == "amorepacific_seongsu":
        print("amorepacific")
        video_sets = [
            {'timestamp': '1719037801588', 'location': 'amorepacific_seoungsu', 'date': '20240622', 'cam': '3'},
        ]
    else:
        assert ValueError

    
    for video_set in video_sets:
        base_path = f"{MAYDRIVE_PATH}/{video_set['location']}/{video_set['date']}/{video_set['cam']}"
        project_file_name = f"{video_set['timestamp']}.mp4"

        print(f"Processing project id {video_set['timestamp']}...")
        video_path = os.path.join(base_path, project_file_name)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp:
            output_video_path = temp.name
            trim_video(video_path, output_video_path, 30)
            tmp_folder = f"/tmp/{video_set['cam']}_{video_set['timestamp']}"
            frame_dir = os.path.join(tmp_folder, "frames")
            if not os.path.exists(frame_dir):
                extract_frames(output_video_path, frame_dir, fps_target=10, max_frames=1500)

            video = daram.Video(output_video_path)

            json_save_path = os.path.join(tmp_folder, "results")
            os.makedirs(json_save_path, exist_ok=True)
            video.data.set_path(json_save_path)

            # DETECTOR
            detect_config = daram.core.detector.base_config.FullBodyDetectorConfig()
            detect_config.model_path = "weights/yolox_x_pretrained.engine"
            detect_config.inference_fps = 10
            detector = daram.core.detector.BodyPoseDetector(
                video, detect_config
                )
            detector.run()
            detector.save(os.path.join(video.data.path, "detection_result_yolox.json"))
            video.data.add_json(
                "detection_result", os.path.join(video.data.path, "detection_result_yolox.json")
            )
            
            ## TRACKER
            if args.tracker == "vispose":
                tracker_config = daram.core.tracker.VisPoseTrackingModuleConfig()
            else:
                tracker_config = daram.core.tracker.VisIOUTrackingModuleConfig()
            tracker_config.feature_extractor_type = "vit-small-ics"
            tracker_config.model_path = "weights/vit-small-ics_v2.engine"
            # tracker_config.model_path = "1635581407_43eph.pt" ###for 3.10.5
            tracker_config.param_preset = "yolox"
            tracker_config.frame_dir = frame_dir
            tracker_config.BYTE_SCORE_HIGH = 0.1
            tracker_config.BATCH_SIZE = 16

            if args.tracker == "vispose":
                tracker = daram.core.tracker.VisPoseTrackingModule(video, tracker_config)
            else:
                tracker = daram.core.tracker.VisIOUTrackingModule(video, tracker_config)
            tracker.run()
            tracker.save_by_frame(os.path.join(json_save_path, f"tracking_result_by_frame_{args.dataset}.json"))
            tracker.save_by_object(
                os.path.join(video.data.path, f"tracking_result_by_object_{args.dataset}.json")
            )

        with open(f"{json_save_path}/tracking_result_by_frame_{args.dataset}.json") as f:
            daram_data = json.load(f)
        
        save_tracker_result_path = f"results/tracker_{args.tracker}/{args.dataset}-every/last_benchmark/data"
        os.makedirs(save_tracker_result_path, exist_ok=True)
        daram_to_mot(daram_data, os.path.join(save_tracker_result_path, f"{video_set['cam']}_{video_set['timestamp']}.txt"))
        print("Saved mot file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("tracker validation inference parser")
    parser.add_argument("--dataset", type=str, default="amorepacific_seongsu")
    parser.add_argument("--tracker", type=str, default="visiou", choices=["vispose", "visiou"])
    # parser.add_argument("--reid_weight", type=str, default="weights/1635581407_43eph.engine")
    args = parser.parse_args()
    
    main(args)