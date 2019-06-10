"""Feature engineering and preprocessing for the Boston housing dataset.

Steps:
    - log-transform highly skewed features (CRIM, LSTAT, DIS)
    - polynomial features for RM and LSTAT (degree 2)
    - drop B (questionable feature)
    - standard scale everything

Returns numpy arrays plus a fitted scaler so the Flask app can re-use it.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# columns we log-transform. small +1 to avoid log(0).
LOG_COLS = ['CRIM', 'LSTAT', 'DIS']

# columns we add poly features for
POLY_COLS = ['RM', 'LSTAT']

# columns to drop. B is racially loaded and not informative anyway.
DROP_COLS = ['B']


def add_log_features(df, cols=LOG_COLS):
    df = df.copy()
    for c in cols:
        df['log_' + c] = np.log1p(df[c])
    return df


def add_poly_features(df, cols=POLY_COLS, degree=2):
    """Add x**2 (and x**3 if degree==3) interactions for the given columns."""
    df = df.copy()
    for c in cols:
        for d in range(2, degree + 1):
            df[c + '_p' + str(d)] = df[c] ** d
    return df


def drop_unwanted(df, cols=DROP_COLS):
    return df.drop(columns=[c for c in cols if c in df.columns])


def build_features(df, poly_degree=2):
    """Apply the full feature engineering pipeline."""
    out = add_log_features(df)
    out = add_poly_features(out, degree=poly_degree)
    out = drop_unwanted(out)
    return out


def fit_scaler(X):
    sc = StandardScaler()
    sc.fit(X)
    return sc


def transform(X, scaler):
    return scaler.transform(X)
