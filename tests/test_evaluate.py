"""Tests unitaires de la logique d'évaluation (sans modèle ni données)."""

from types import SimpleNamespace

import numpy as np
import pytest

from ecosort.training.evaluate import best_threshold, operating_point, per_class


def make_box():
    px = np.linspace(0, 1, 5)  # seuils 0, 0.25, 0.5, 0.75, 1
    f1 = np.array([[0.1, 0.6, 0.8, 0.4, 0.0], [0.1, 0.4, 0.6, 0.2, 0.0]])
    p = np.array([[0.2, 0.5, 0.8, 0.9, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0]])
    r = np.array([[1.0, 0.8, 0.7, 0.3, 0.0], [1.0, 0.6, 0.5, 0.2, 0.0]])
    results = [(0.9, 0.5, 0.6, 0.4), (0.3, 0.1, 0.2, 0.1)]
    return SimpleNamespace(
        px=px, f1_curve=f1, p_curve=p, r_curve=r,
        ap_class_index=np.array([0, 2]),
        class_result=lambda i: results[i],
    )


def test_best_threshold_maximizes_mean_f1():
    conf, f1 = best_threshold(make_box())
    assert conf == pytest.approx(0.5)
    assert f1 == pytest.approx(0.7)


def test_best_threshold_empty_curve_falls_back():
    box = SimpleNamespace(f1_curve=np.empty((0, 0)), px=np.empty(0))
    assert best_threshold(box) == (0.25, 0.0)


def test_operating_point_reads_nearest_threshold():
    op = operating_point(make_box(), conf=0.49)
    assert op["precision"] == pytest.approx(0.7)
    assert op["recall"] == pytest.approx(0.6)
    assert op["f1"] == pytest.approx(0.7)


def test_per_class_maps_indices_and_zero_fills_absent_classes():
    out = per_class(make_box(), ["plastic", "paper", "glass"])
    assert out["plastic"]["mAP50"] == 0.6
    assert out["glass"]["recall"] == 0.1
    assert out["paper"] == {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50_95": 0.0}