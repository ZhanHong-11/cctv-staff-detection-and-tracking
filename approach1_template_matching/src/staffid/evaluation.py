"""Evaluation helpers: matching boxes to ground truth, AUC, frame-level metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise IoU of two (N, 4) xyxy arrays."""
    ix1, iy1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    ix2, iy2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


def label_boxes_with_gt(boxes: pd.DataFrame, gt: pd.DataFrame, min_iou: float = 0.5) -> pd.DataFrame:
    """Add is_staff (box overlaps the GT staff box with IoU >= min_iou) and certain columns."""
    m = boxes.merge(gt[["frame", "staff_present", "x1", "y1", "x2", "y2", "certain"]],
                    on="frame", how="left", suffixes=("", "_gt"))
    a = m[["x1", "y1", "x2", "y2"]].to_numpy(float)
    b = m[["x1_gt", "y1_gt", "x2_gt", "y2_gt"]].to_numpy(float)
    ov = np.nan_to_num(iou(a, b))
    m["iou_gt"] = ov
    m["is_staff"] = ((m.staff_present == 1) & (ov >= min_iou)).astype(int)
    return m.drop(columns=["x1_gt", "y1_gt", "x2_gt", "y2_gt", "staff_present"])


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive scores higher than a random negative (ties count half)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def frame_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Precision / recall / F1 / accuracy for binary per-frame labels."""
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "accuracy": round((tp + tn) / max(len(pred), 1), 4)}
