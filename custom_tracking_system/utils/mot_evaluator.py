"""
MOT Evaluator -- MOTA / IDF1 / precision / recall via py-motmetrics.

Reads the simple per-camera CSV files produced by:
  - carla_bridge/server.py --eval-output <dir>   -> gt_<CAM_ID>.txt
  - main.py --source bridge --eval-output <dir>  -> pred_<CAM_ID>.txt

File format (one detection per line, no header):
    frame,id,x1,y1,x2,y2,class

This module does NOT depend on motmetrics' file-format loaders -- it builds
a MOTAccumulator directly from per-frame ID/box arrays, which keeps the
on-disk format simple and avoids column-count mismatches.
"""

import csv
import logging
from collections import defaultdict

import numpy as np
import motmetrics as mm

logger = logging.getLogger(__name__)


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray, max_iou: float = 0.5) -> np.ndarray:
    """Distance matrix (1 - IoU) between gt and pred boxes, both [x,y,w,h].

    Entries with IoU below (1 - max_iou) are set to NaN (no match allowed),
    matching the convention expected by mm.MOTAccumulator.update().

    Implemented directly (not via mm.distances.iou_matrix) to avoid relying
    on motmetrics internals that broke under numpy>=2.0 (np.asfarray removal).
    """
    n, m = len(gt_boxes), len(pred_boxes)
    dists = np.full((n, m), np.nan)
    if n == 0 or m == 0:
        return dists

    gx1, gy1 = gt_boxes[:, 0], gt_boxes[:, 1]
    gx2, gy2 = gx1 + gt_boxes[:, 2], gy1 + gt_boxes[:, 3]
    px1, py1 = pred_boxes[:, 0], pred_boxes[:, 1]
    px2, py2 = px1 + pred_boxes[:, 2], py1 + pred_boxes[:, 3]

    for i in range(n):
        ix1 = np.maximum(gx1[i], px1)
        iy1 = np.maximum(gy1[i], py1)
        ix2 = np.minimum(gx2[i], px2)
        iy2 = np.minimum(gy2[i], py2)

        iw = np.clip(ix2 - ix1, 0, None)
        ih = np.clip(iy2 - iy1, 0, None)
        inter = iw * ih

        area_g = gt_boxes[i, 2] * gt_boxes[i, 3]
        area_p = pred_boxes[:, 2] * pred_boxes[:, 3]
        union = area_g + area_p - inter

        iou = np.divide(inter, union, out=np.zeros_like(union), where=union > 0)
        dist = 1.0 - iou
        dist[iou < (1.0 - max_iou)] = np.nan
        dists[i, :] = dist

    return dists


def load_tracks(path: str) -> dict:
    """Load a frame,id,x1,y1,x2,y2[,class] CSV file.

    Returns:
        {frame_number: {'ids': [int, ...], 'boxes': np.ndarray Nx4 [x,y,w,h]}}
    """
    data = defaultdict(lambda: {'ids': [], 'boxes': []})
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame = int(row[0])
            obj_id = int(row[1])
            x1, y1, x2, y2 = (float(v) for v in row[2:6])
            data[frame]['ids'].append(obj_id)
            data[frame]['boxes'].append([x1, y1, x2 - x1, y2 - y1])

    for frame in data:
        data[frame]['boxes'] = np.array(data[frame]['boxes'], dtype=float)

    return data


class MOTEvaluator:
    """Compute MOTA/IDF1/precision/recall for one or more cameras."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate_camera(self, gt_path: str, pred_path: str) -> mm.MOTAccumulator:
        """Build a MOTAccumulator by matching gt vs. pred boxes frame-by-frame."""
        gt = load_tracks(gt_path)
        pred = load_tracks(pred_path)

        acc = mm.MOTAccumulator(auto_id=True)
        empty = np.empty((0, 4))

        for frame in sorted(set(gt) | set(pred)):
            g = gt.get(frame, {'ids': [], 'boxes': empty})
            p = pred.get(frame, {'ids': [], 'boxes': empty})
            dists = iou_matrix(g['boxes'], p['boxes'], max_iou=self.iou_threshold)
            acc.update(g['ids'], p['ids'], dists)

        return acc

    @staticmethod
    def summarize(accumulators: dict, metrics=None):
        """accumulators: {name: MOTAccumulator} -> pandas.DataFrame summary."""
        mh = mm.metrics.create()
        metrics = metrics or mm.metrics.motchallenge_metrics
        return mh.compute_many(
            list(accumulators.values()),
            metrics=metrics,
            names=list(accumulators.keys()),
            generate_overall=True,
        )

    @staticmethod
    def render_summary(summary) -> str:
        mh = mm.metrics.create()
        return mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )
