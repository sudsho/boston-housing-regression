"""Load the housing dataset.

The classic Boston housing dataset was removed from scikit-learn (>= 1.2) for
ethical reasons, so ``load_boston`` no longer exists in modern installs. To
keep this project runnable end to end and fully offline, ``load_data`` tries a
few sources in order and falls back to a deterministic synthetic dataset that
has the same 13 Boston feature columns, so the rest of the pipeline
(preprocessing, training, the Flask API) works unchanged.

Load order:
    1. a bundled CSV at ``data/boston.csv`` (drop the real data here if you
       have it) with the 13 feature columns plus ``MEDV``
    2. legacy ``sklearn.datasets.load_boston`` (only on old scikit-learn)
    3. a synthetic Boston-shaped regression dataset (always available, offline)
"""
import os

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]
TARGET_NAME = 'MEDV'

_BUNDLED_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'boston.csv'
)


def _from_csv(path):
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_NAME].astype(float).rename(TARGET_NAME)
    return X, y


def _from_sklearn():
    # Only importable on old scikit-learn (< 1.2). Raises on modern installs.
    from sklearn.datasets import load_boston
    raw = load_boston()
    X = pd.DataFrame(raw.data, columns=FEATURE_NAMES)
    y = pd.Series(raw.target, name=TARGET_NAME)
    return X, y


def make_synthetic(n_samples=506, seed=42):
    """Deterministic, offline, Boston-shaped synthetic dataset.

    Columns match the real Boston feature names and stay in plausible positive
    ranges (the log-transformed columns CRIM/DIS/LSTAT and RM are kept > 0 so
    the feature engineering pipeline behaves). The target is a noisy nonlinear
    function of a few features so the regressors have real signal to learn and
    RandomForest can beat the linear models, mirroring the original project.
    """
    rng = np.random.RandomState(seed)

    crim = np.abs(rng.lognormal(mean=-0.5, sigma=1.2, size=n_samples))
    zn = rng.choice([0.0, 12.5, 25.0, 40.0], size=n_samples, p=[0.6, 0.2, 0.1, 0.1])
    indus = rng.uniform(1.0, 27.0, size=n_samples)
    chas = rng.binomial(1, 0.07, size=n_samples).astype(float)
    nox = rng.uniform(0.38, 0.87, size=n_samples)
    rm = np.clip(rng.normal(6.3, 0.7, size=n_samples), 3.5, 8.8)
    age = rng.uniform(2.0, 100.0, size=n_samples)
    dis = np.abs(rng.lognormal(mean=1.1, sigma=0.5, size=n_samples)) + 0.5
    rad = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 24], size=n_samples).astype(float)
    tax = rng.uniform(180.0, 711.0, size=n_samples)
    ptratio = rng.uniform(12.6, 22.0, size=n_samples)
    b = np.clip(rng.normal(356.0, 90.0, size=n_samples), 0.3, 396.9)
    lstat = np.clip(np.abs(rng.lognormal(mean=2.2, sigma=0.5, size=n_samples)), 1.0, 38.0)

    X = pd.DataFrame({
        'CRIM': crim, 'ZN': zn, 'INDUS': indus, 'CHAS': chas,
        'NOX': nox, 'RM': rm, 'AGE': age, 'DIS': dis,
        'RAD': rad, 'TAX': tax, 'PTRATIO': ptratio, 'B': b, 'LSTAT': lstat,
    })[FEATURE_NAMES]

    # Target: rooms push price up, poverty / crime / pollution push it down,
    # plus a nonlinear rm*lstat interaction and gaussian noise.
    medv = (
        12.0
        + 6.0 * (rm - 6.0)
        - 0.55 * lstat
        - 1.2 * np.log1p(crim)
        - 9.0 * (nox - 0.5)
        - 0.02 * (ptratio - 15.0) * 3.0
        + 0.15 * (rm - 6.0) * (20.0 - lstat)
        + 3.0 * chas
        + rng.normal(0.0, 2.5, size=n_samples)
    )
    y = pd.Series(np.clip(medv, 5.0, 50.0), name=TARGET_NAME)
    return X, y


def load_data():
    """Returns features dataframe X and target series y.

    Uses the first source that works: bundled CSV, legacy sklearn, then the
    offline synthetic fallback.
    """
    if os.path.exists(_BUNDLED_CSV):
        try:
            return _from_csv(_BUNDLED_CSV)
        except Exception:
            pass
    try:
        return _from_sklearn()
    except Exception:
        return make_synthetic()


if __name__ == '__main__':
    X, y = load_data()
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print(X.head())
