import os
import sys
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    
    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_FOLDER', default="/user/gt", help='Location of GT data')
    parser.add_argument('--TRACKERS_FOLDER', default="/user/results/tracker_visiou", help='Trackers location')
    parser.add_argument('--BENCHMARKS', default=["MOT17", "MOT20", "DanceTrack"], help='List of benchmarks to evaluate')
    parser.add_argument('--SPLIT_TO_EVAL', default=["train", "train", "val"], help='List of splits to evaluate for each benchmark')
    args = parser.parse_args()

    # Ensure the number of benchmarks matches the number of splits
    if len(args.BENCHMARKS) != len(args.SPLIT_TO_EVAL):
        raise ValueError("Number of benchmarks must match the number of splits provided.")

    # Iterate over each benchmark and split combination
    for benchmark, split in zip(args.BENCHMARKS, args.SPLIT_TO_EVAL):
        print(f"Evaluating Benchmark: {benchmark}, Split: {split}")

        # Update config for each benchmark and split
        config['GT_FOLDER'] = args.GT_FOLDER
        config['TRACKERS_FOLDER'] = args.TRACKERS_FOLDER
        config['BENCHMARK'] = benchmark
        config['SPLIT_TO_EVAL'] = split

        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

        # Run evaluation
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(metrics_config))
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)

        print(f"Finished Evaluating Benchmark: {benchmark}, Split: {split}\n")