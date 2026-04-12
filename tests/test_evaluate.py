import json

import numpy as np
import pytest

from ates import evaluate


def _make_rasters(predicted_vals, truth_vals, shape=(10, 10)):
    """Fill rasters with given flat values over the valid region."""
    predicted = np.full(shape, -9999, dtype=np.int16)
    truth = np.full(shape, -9999, dtype=np.int16)
    predicted[1:-1, 1:-1] = predicted_vals
    truth[1:-1, 1:-1] = truth_vals
    return predicted, truth


# ---------------------------------------------------------------------------
# report — print path
# ---------------------------------------------------------------------------

def test_report_prints(capsys):
    predicted, truth = _make_rasters(1, 1)
    evaluate.report(predicted, truth)
    out = capsys.readouterr().out
    assert "Simple" in out
    assert "precision" in out


def test_report_ignores_nodata(capsys):
    predicted, truth = _make_rasters(2, 2)
    # Border pixels are -9999; report should still work on interior pixels
    evaluate.report(predicted, truth)
    out = capsys.readouterr().out
    assert "Challenging" in out


# ---------------------------------------------------------------------------
# report — save path
# ---------------------------------------------------------------------------

def test_report_saves_json(tmp_path):
    predicted, truth = _make_rasters(1, 1)
    evaluate.report(predicted, truth, save_dir=tmp_path)
    result_file = tmp_path / "results.json"
    assert result_file.exists()
    data = json.loads(result_file.read_text())
    assert "Simple" in data


def test_report_save_does_not_print(capsys, tmp_path):
    predicted, truth = _make_rasters(1, 1)
    evaluate.report(predicted, truth, save_dir=tmp_path)
    assert capsys.readouterr().out == ""


def test_report_json_contains_f1(tmp_path):
    predicted, truth = _make_rasters(1, 1)
    evaluate.report(predicted, truth, save_dir=tmp_path)
    data = json.loads((tmp_path / "results.json").read_text())
    assert "f1-score" in data["Simple"]


# ---------------------------------------------------------------------------
# plot functions — just assert they don't raise
# ---------------------------------------------------------------------------

def test_plot_confusion_matrix_runs(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    predicted, truth = _make_rasters(1, 1)
    evaluate.plot_confusion_matrix(predicted, truth)


def test_plot_side_by_side_runs(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    predicted, truth = _make_rasters(2, 3)
    evaluate.plot_side_by_side(predicted, truth, predicted_title="Model")


def test_plot_side_by_side_custom_titles(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    predicted, truth = _make_rasters(1, 2)
    # Should not raise with custom titles
    evaluate.plot_side_by_side(predicted, truth, predicted_title="A", truth_title="B")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_report_all_nodata(capsys):
    predicted = np.full((5, 5), -9999, dtype=np.int16)
    truth = np.full((5, 5), -9999, dtype=np.int16)
    # classification_report raises if there are no samples — that's acceptable
    with pytest.raises(ValueError):
        evaluate.report(predicted, truth)


def test_report_mixed_classes(capsys):
    shape = (10, 10)
    predicted = np.full(shape, -9999, dtype=np.int16)
    truth = np.full(shape, -9999, dtype=np.int16)
    predicted[0, :] = 1
    predicted[1, :] = 2
    predicted[2, :] = 3
    truth[0, :] = 1
    truth[1, :] = 2
    truth[2, :] = 3
    evaluate.report(predicted, truth)
    out = capsys.readouterr().out
    assert "Simple" in out
    assert "Challenging" in out
    assert "Complex" in out
