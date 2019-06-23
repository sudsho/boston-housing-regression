"""Tests for evaluate module."""
import numpy as np

from src.evaluate import metrics, pretty


def test_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = metrics(y, y)
    assert abs(m['mae']) < 1e-9
    assert abs(m['rmse']) < 1e-9
    assert abs(m['r2'] - 1.0) < 1e-9


def test_metrics_off_by_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = y + 1.0
    m = metrics(y, yhat)
    assert abs(m['mae'] - 1.0) < 1e-9
    assert abs(m['rmse'] - 1.0) < 1e-9


def test_pretty_formats_three_metrics():
    s = pretty({'mae': 1.0, 'rmse': 2.0, 'r2': 0.5})
    assert 'MAE=1.000' in s
    assert 'RMSE=2.000' in s
    assert 'R2=0.500' in s
