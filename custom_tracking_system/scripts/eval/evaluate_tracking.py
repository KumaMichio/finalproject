#!/usr/bin/env python3
"""
Ground Truth Evaluation -- MOTA / IDF1 / precision / recall for ByteTrack.

Pairs gt_<CAM_ID>.txt (ground truth, from CARLA actors) with
pred_<CAM_ID>.txt (AI pipeline output) in --eval-dir and reports
MOT Challenge metrics per camera + overall.

Usage:
    1) Run the AI pipeline with CARLA, writing both ground truth and
       predictions (--source carla has direct CARLA world access):
         python main.py --source carla --eval-output data/eval --max-frames 1000

    2) Evaluate:
         python evaluate_tracking.py --eval-dir data/eval \\
             --output docs/assets/mot_eval/summary.csv --chart docs/assets/mot_eval/mot_metrics.png

(Historical --source bridge path: carla_bridge/server.py --eval-output data/eval
 writes gt_<CAM_ID>.txt, main.py --source bridge --eval-output data/eval writes
 pred_<CAM_ID>.txt.)
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.mot_evaluator import MOTEvaluator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='MOT evaluation (MOTA/IDF1) for ByteTrack')
    parser.add_argument('--eval-dir', default='data/eval',
                        help='Directory containing gt_<CAM_ID>.txt / pred_<CAM_ID>.txt')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--output', default=None, help='Optional path to save summary as CSV')
    parser.add_argument('--chart', default=None,
                        help='Optional path to save a MOTA/IDF1/Precision/Recall bar chart (PNG)')
    args = parser.parse_args()

    evaluator = MOTEvaluator(iou_threshold=args.iou_threshold)

    accumulators = {}
    for gt_path in sorted(glob.glob(os.path.join(args.eval_dir, 'gt_*.txt'))):
        cam_id = os.path.basename(gt_path)[len('gt_'):-len('.txt')]
        pred_path = os.path.join(args.eval_dir, 'pred_{}.txt'.format(cam_id))
        if not os.path.exists(pred_path):
            print('[SKIP] {}: missing predictions ({})'.format(cam_id, pred_path))
            continue
        accumulators[cam_id] = evaluator.evaluate_camera(gt_path, pred_path)

    if not accumulators:
        print('No gt_*.txt / pred_*.txt pairs found in {}'.format(args.eval_dir))
        sys.exit(1)

    summary = evaluator.summarize(accumulators)
    print(evaluator.render_summary(summary))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        summary.to_csv(args.output)
        print('\nSaved -> {}'.format(args.output))

    if args.chart:
        save_chart(summary, args.chart)
        print('Saved -> {}'.format(args.chart))


def save_chart(summary, path):
    """Bar chart of MOTA/IDF1/Precision/Recall per camera (incl. OVERALL)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ['mota', 'idf1', 'precision', 'recall']
    labels = ['MOTA', 'IDF1', 'Precision', 'Recall']
    cameras = list(summary.index)

    x = np.arange(len(cameras))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(6, len(cameras) * 1.5), 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = summary[metric].values * 100
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(cameras)
    ax.set_ylabel('%')
    ax.set_ylim(0, 100)
    ax.set_title('MOT Evaluation -- ByteTrack')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
