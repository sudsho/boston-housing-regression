"""Tests for the preprocess module."""
import numpy as np
import pandas as pd

from src.preprocess import (
    add_log_features,
    add_poly_features,
    drop_unwanted,
    build_features,
)


def make_df():
    return pd.DataFrame({
        'CRIM':  [0.1, 0.5, 1.0],
        'ZN':    [0.0, 0.0, 0.0],
        'INDUS': [5.0, 6.0, 7.0],
        'CHAS':  [0, 0, 1],
        'NOX':   [0.5, 0.5, 0.5],
        'RM':    [6.0, 6.5, 7.0],
        'AGE':   [50.0, 60.0, 70.0],
        'DIS':   [2.0, 3.0, 4.0],
        'RAD':   [1, 2, 3],
        'TAX':   [200, 300, 400],
        'PTRATIO': [15.0, 16.0, 17.0],
        'B':     [350.0, 360.0, 370.0],
        'LSTAT': [5.0, 10.0, 15.0],
    })


def test_log_features_added():
    df = make_df()
    out = add_log_features(df)
    assert 'log_CRIM' in out.columns
    assert 'log_LSTAT' in out.columns
    assert 'log_DIS' in out.columns
    # log1p(0.1) is ~0.0953
    np.testing.assert_almost_equal(out['log_CRIM'].iloc[0], np.log1p(0.1), decimal=6)


def test_poly_features():
    df = make_df()
    out = add_poly_features(df, cols=['RM'], degree=2)
    assert 'RM_p2' in out.columns
    np.testing.assert_almost_equal(out['RM_p2'].iloc[0], 36.0)


def test_drop_unwanted():
    df = make_df()
    out = drop_unwanted(df)
    assert 'B' not in out.columns


def test_build_features_full_pipeline():
    df = make_df()
    out = build_features(df)
    assert 'B' not in out.columns
    assert 'log_CRIM' in out.columns
    assert 'RM_p2' in out.columns
    assert 'RM_x_LSTAT' in out.columns
    # row count must be preserved
    assert len(out) == len(df)
