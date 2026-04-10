"""Evaluation metrics for ATES classification models.

Compares a predicted ATES raster against a ground-truth raster and returns
per-class precision, recall, F1, and macro F1 — matching the metrics reported
in Vors et al. (2024).

All functions ignore nodata pixels (value -9999) in both arrays.
Classes are 1 (Simple) through 4 (Extreme).
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Build a 4×4 confusion matrix for ATES classes 1–4.

    Rows = true class, columns = predicted class.
    Pixels where either array is nodata (-9999) are excluded.

    Returns:
        np.ndarray of shape (4, 4), dtype int64.
    """
    mask = (truth != -9999) & (predicted != -9999)
    t = truth[mask].astype(np.int16)
    p = predicted[mask].astype(np.int16)

    mat = np.zeros((4, 4), dtype=np.int64)
    for i, true_cls in enumerate(range(1, 5)):
        for j, pred_cls in enumerate(range(1, 5)):
            mat[i, j] = int(np.sum((t == true_cls) & (p == pred_cls)))
    return mat


def compute_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Compute per-class and macro F1 scores.

    Args:
        predicted: int16 ATES raster, nodata = -9999.
        truth:     int16 ground-truth raster, nodata = -9999.

    Returns:
        {
            "class_f1":        {1: float, 2: float, 3: float, 4: float},
            "macro_f1":        float,
            "precision":       {1: float, ...},
            "recall":          {1: float, ...},
            "confusion_matrix": np.ndarray (4×4),
        }
    """
    mat = confusion_matrix(predicted, truth)

    class_f1: dict[int, float] = {}
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}

    for i, cls in enumerate(range(1, 5)):
        tp = int(mat[i, i])
        fp = int(mat[:, i].sum()) - tp   # predicted cls but not true cls
        fn = int(mat[i, :].sum()) - tp   # true cls but not predicted cls

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision[cls] = p
        recall[cls] = r
        class_f1[cls] = f1

    macro_f1 = float(np.mean(list(class_f1.values())))

    return {
        "class_f1": class_f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": mat,
    }
