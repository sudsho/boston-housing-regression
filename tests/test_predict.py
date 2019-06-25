"""Tests for predict and the flask /predict route."""
import json
import os
import tempfile

import joblib
import pytest

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_NAMES, load_data
from src.preprocess import build_features
from src import predict as predict_mod
import app as app_module


@pytest.fixture
def trained_artifacts():
    """Train a tiny ridge model on the real data and dump artifacts to a tmp dir.

    We then point the predict module at the tmp dir.
    """
    X, y = load_data()
    X_fe = build_features(X)
    sc = StandardScaler().fit(X_fe.values)
    X_s = sc.transform(X_fe.values)

    model = Ridge(alpha=1.0)
    model.fit(X_s, y)

    tmp = tempfile.mkdtemp()
    joblib.dump(model, os.path.join(tmp, 'model.pkl'))
    joblib.dump(sc, os.path.join(tmp, 'scaler.pkl'))
    joblib.dump(list(X_fe.columns), os.path.join(tmp, 'feature_columns.pkl'))

    # reset cached module-level model so it picks up the tmp one
    predict_mod._MODEL = None
    predict_mod._SCALER = None
    predict_mod._COLUMNS = None

    yield tmp

    predict_mod._MODEL = None
    predict_mod._SCALER = None
    predict_mod._COLUMNS = None


def sample_record():
    return {
        'CRIM': 0.1, 'ZN': 0.0, 'INDUS': 5.0, 'CHAS': 0,
        'NOX': 0.5, 'RM': 6.0, 'AGE': 50.0, 'DIS': 4.0,
        'RAD': 1, 'TAX': 296, 'PTRATIO': 15.3,
        'B': 396.9, 'LSTAT': 5.0,
    }


def test_predict_one_returns_float(trained_artifacts):
    p = predict_mod.predict_one(sample_record(), model_dir=trained_artifacts)
    assert isinstance(p, float)
    # Boston MEDV is in thousands of $; reasonable range is ~5 to 50
    assert 0 < p < 80


def test_flask_health():
    client = app_module.app.test_client()
    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_json()['status'] == 'ok'


def test_flask_predict_missing_field(trained_artifacts):
    client = app_module.app.test_client()
    bad = sample_record()
    bad.pop('LSTAT')
    r = client.post('/predict', data=json.dumps(bad), content_type='application/json')
    assert r.status_code == 400
    body = r.get_json()
    assert 'LSTAT' in body['missing']


def test_flask_predict_ok(trained_artifacts, monkeypatch):
    # make app's predict_one use the tmp model dir
    monkeypatch.setattr(
        app_module, 'predict_one',
        lambda rec: predict_mod.predict_one(rec, model_dir=trained_artifacts)
    )
    client = app_module.app.test_client()
    r = client.post('/predict', data=json.dumps(sample_record()),
                    content_type='application/json')
    assert r.status_code == 200
    body = r.get_json()
    assert 'medv' in body
    assert isinstance(body['medv'], float)
