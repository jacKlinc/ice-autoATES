"""Evaluation utilities for ATES classification models.

All functions ignore nodata pixels (value -9999) in both arrays.
Classes are 1 (Simple), 2 (Challenging), 3 (Complex).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ATES_LABELS = ["Simple", "Challenging", "Complex"]
ATES_CLASSES = [1, 2, 3]
_ATES_COLOURS = ["#00b400", "#1e50ff", "#282828"]
_CMAP = ListedColormap(_ATES_COLOURS)
_NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], _CMAP.N)


def report(
    predicted: np.ndarray,
    truth: np.ndarray,
    save_dir: Path | str | None = None,
) -> None:
    """Print sklearn classification report for ATES classes 1–3.

    Args:
        predicted: int16 ATES raster, nodata = -9999. Extreme (4) should be
                   collapsed to Complex (3) before calling when validating
                   against ATES v1 ground truth.
        truth:     int16 ground-truth raster, nodata = -9999.
        save_dir:  optional directory in which to write results.json.
    """
    mask = (truth != -9999) & (predicted != -9999)
    results = classification_report(
        truth[mask],
        predicted[mask],
        labels=ATES_CLASSES,
        target_names=ATES_LABELS,
        output_dict=save_dir is not None,
    )
    if save_dir is not None:
        (Path(save_dir) / "results.json").write_text(json.dumps(results, indent=2))
    else:
        print(results)


def plot_confusion_matrix(predicted: np.ndarray, truth: np.ndarray) -> None:
    """Plot confusion matrix comparing predicted vs. ground-truth ATES rasters.

    Args:
        predicted: int16 ATES raster, nodata = -9999.
        truth:     int16 ground-truth raster, nodata = -9999.
    """
    mask = (truth != -9999) & (predicted != -9999)
    cm = confusion_matrix(truth[mask], predicted[mask], labels=ATES_CLASSES)

    disp = ConfusionMatrixDisplay(cm, display_labels=ATES_LABELS)
    _, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix — simple model vs. Avalanche Canada (v1)")
    plt.tight_layout()
    plt.show()


def plot_side_by_side(
    predicted: np.ndarray,
    truth: np.ndarray,
    predicted_title: str = "Predicted",
    truth_title: str = "Ground truth (Avalanche Canada v1)",
) -> None:
    """Plot predicted and ground-truth ATES rasters side by side.

    Args:
        predicted: int16 ATES raster, nodata = -9999.
        truth:     int16 ground-truth raster, nodata = -9999.
        predicted_title: title for the predicted panel.
        truth_title:     title for the ground-truth panel.
    """

    def _show(ax: plt.Axes, arr: np.ndarray, title: str) -> None:
        disp = arr.astype(float)
        disp[disp == -9999] = np.nan
        ax.imshow(disp, cmap=_CMAP, norm=_NORM, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    # Mask predicted to ground-truth coverage so both panels share the same footprint
    predicted_masked = np.where(truth != -9999, predicted, -9999).astype(np.int16)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _show(axes[0], predicted_masked, predicted_title)
    _show(axes[1], truth, truth_title)

    legend = [Patch(color=c, label=l) for c, l in zip(_ATES_COLOURS, ATES_LABELS)]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()
