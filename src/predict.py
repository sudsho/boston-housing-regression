"""Load a trained model and run a prediction on a single record."""
import os

import joblib
import pandas as pd

from src.preprocess import build_features


_MODEL = None
_SCALER = None
_COLUMNS = None


def _load(model_dir='models'):
    global _MODEL, _SCALER, _COLUMNS
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(model_dir, 'model.pkl'))
    if _SCALER is None:
        _SCALER = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    if _COLUMNS is None:
        _COLUMNS = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
    return _MODEL, _SCALER, _COLUMNS


def predict_one(record, model_dir='models'):
    """record: dict mapping the 13 raw feature names -> float values."""
    model, scaler, columns = _load(model_dir)

    df = pd.DataFrame([record])
    df = build_features(df)

    # ensure column order matches what the scaler was fit on
    for c in columns:
        if c not in df.columns:
            df[c] = 0.0
    df = df[columns]

    arr = scaler.transform(df.values)
    pred = model.predict(arr)
    return float(pred[0])
