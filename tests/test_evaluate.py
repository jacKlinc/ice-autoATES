from __future__ import annotations

import numpy as np
import pytest

from ates.evaluate import compute_metrics, confusion_matrix


def _r(*values: int) -> np.ndarray:
    """Build a flat int16 raster from literal values. Use -9999 for nodata."""
    return np.array(values, dtype=np.int16)


class TestConfusionMatrix:
    """Unit tests for evaluate.confusion_matrix."""

    def test_shape(self):
        """Output is always a 4×4 matrix regardless of input size."""
        mat = confusion_matrix(_r(1, 2, 3, 4), _r(1, 2, 3, 4))
        assert mat.shape == (4, 4)

    def test_perfect(self):
        """Perfect predictions produce an identity matrix."""
        mat = confusion_matrix(_r(1, 2, 3, 4), _r(1, 2, 3, 4))
        np.testing.assert_array_equal(mat, np.eye(4, dtype=np.int64))

    def test_all_wrong(self):
        """All class-1 pixels predicted as class-2 fill off-diagonal only."""
        mat = confusion_matrix(_r(2, 2, 2, 2), _r(1, 1, 1, 1))
        assert mat[0, 1] == 4
        assert mat[0, 0] == 0
        assert mat.sum() == 4

    @pytest.mark.parametrize("nodata_in,valid_pixels", [
        ("truth",     2),   # one nodata in truth
        ("predicted", 2),   # one nodata in pred
        ("both",      0),   # all nodata
    ])
    def test_nodata_excluded(self, nodata_in, valid_pixels):
        """Pixels where either array is -9999 are excluded from the count."""
        if nodata_in == "truth":
            truth, pred = _r(1, -9999, 3), _r(1, 2, 3)
        elif nodata_in == "predicted":
            truth, pred = _r(1, 2, 3), _r(1, -9999, 3)
        else:
            truth, pred = _r(-9999, -9999), _r(-9999, -9999)
        assert confusion_matrix(pred, truth).sum() == valid_pixels


class TestComputeMetrics:
    """Unit tests for evaluate.compute_metrics."""

    def test_perfect_prediction(self):
        """All metrics are 1.0 when predicted == truth for every class."""
        arr = _r(1, 2, 3, 4, 1, 2, 3, 4)
        m = compute_metrics(arr, arr)
        assert m["macro_f1"] == pytest.approx(1.0)
        for cls in range(1, 5):
            assert m["class_f1"][cls] == pytest.approx(1.0)
            assert m["precision"][cls] == pytest.approx(1.0)
            assert m["recall"][cls] == pytest.approx(1.0)

    def test_all_wrong(self):
        """F1 is 0.0 for all classes when no prediction is correct."""
        m = compute_metrics(_r(2, 2, 2, 2), _r(1, 1, 1, 1))
        assert m["class_f1"][1] == pytest.approx(0.0)
        assert m["class_f1"][2] == pytest.approx(0.0)
        assert m["macro_f1"] == pytest.approx(0.0)

    def test_macro_is_mean_of_class_f1(self):
        """Macro F1 is the unweighted mean of the four per-class F1 scores."""
        arr = _r(1, 2, 3, 4)
        m = compute_metrics(arr, arr)
        assert m["macro_f1"] == pytest.approx(sum(m["class_f1"].values()) / 4)

    def test_missing_class_gets_zero_f1(self):
        """A class absent from the truth raster receives F1 = 0.0."""
        arr = _r(1, 2, 3, 1, 2, 3)   # class 4 absent
        m = compute_metrics(arr, arr)
        assert m["class_f1"][4] == pytest.approx(0.0)
        for cls in [1, 2, 3]:
            assert m["class_f1"][cls] == pytest.approx(1.0)

    def test_partial_overlap(self):
        """Precision, recall, and F1 are correct for partial misclassification."""
        # 2 correct class-1, 1 misclassified as class-2
        m = compute_metrics(_r(1, 1, 2), _r(1, 1, 1))
        assert m["precision"][1] == pytest.approx(1.0)
        assert m["recall"][1] == pytest.approx(2 / 3)
        expected = 2 * (1.0 * 2 / 3) / (1.0 + 2 / 3)
        assert m["class_f1"][1] == pytest.approx(expected)

    def test_returns_confusion_matrix(self):
        """Return dict includes the raw 4×4 confusion matrix."""
        m = compute_metrics(_r(1, 2), _r(1, 2))
        assert "confusion_matrix" in m
        assert m["confusion_matrix"].shape == (4, 4)

    @pytest.mark.parametrize("cls", [1, 2, 3, 4])
    def test_nodata_excluded_per_class(self, cls):
        """A nodata pixel in truth is excluded; the remaining correct pixel scores F1=1."""
        m = compute_metrics(_r(cls, cls), _r(cls, -9999))
        assert m["class_f1"][cls] == pytest.approx(1.0)
