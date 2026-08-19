"""Offline end-to-end smoke test for the housing regression project.

Runs with no network access. It:
    1. loads the dataset (bundled CSV / legacy sklearn / synthetic fallback)
    2. trains 3 regressors on a fixed split and prints RMSE / R2 for each
    3. dumps model + scaler artifacts (so the serving path has something to load)
    4. exercises the predict helper on a single held-out style sample
    5. exercises the Flask /health and /predict routes via the test client

Exit code is non-zero if any stage fails, so `make smoke` / CI can gate on it.
"""
import json
import os
import sys

# make sure the repo root is importable when run as `python scripts/smoke.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.model_selection import train_test_split

from src.data import load_data, make_synthetic
from src.preprocess import build_features, fit_scaler, transform
from src.model import build_model
from src.evaluate import metrics
from src.train import train, load_config
from src import predict as predict_mod
import app as app_module


def _data_source():
    """Report which loader path load_data() actually used."""
    from src import data as data_mod
    if os.path.exists(data_mod._BUNDLED_CSV):
        return 'bundled CSV (data/boston.csv)'
    try:
        from sklearn.datasets import load_boston  # noqa: F401
        return 'legacy sklearn load_boston'
    except Exception:
        return 'synthetic Boston-shaped fallback (make_synthetic, offline)'


def main():
    print('== boston-housing-regression offline smoke ==')

    # 1. load data
    X, y = load_data()
    print('[1] loaded data: X=%s y=%s  source=%s' % (X.shape, y.shape, _data_source()))

    # 2. train 3 regressors on one fixed split, print metrics
    X_fe = build_features(X, poly_degree=2)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fe, y, test_size=0.2, random_state=42
    )
    scaler = fit_scaler(X_tr)
    X_tr_s = transform(X_tr, scaler)
    X_te_s = transform(X_te, scaler)

    print('[2] training regressors (RMSE / R2 on held-out 20%):')
    for name, cfg in [
        ('linear', {'type': 'linear'}),
        ('ridge', {'type': 'ridge', 'ridge_alpha': 1.0}),
        ('rf', {'type': 'rf', 'rf_n_estimators': 200, 'rf_max_depth': 10}),
    ]:
        model = build_model(cfg)
        model.fit(X_tr_s, y_tr)
        m = metrics(y_te, model.predict(X_te_s))
        print('      %-8s RMSE=%.3f  R2=%.3f' % (name, m['rmse'], m['r2']))

    # 3. dump artifacts via the real training entrypoint
    cfg = load_config(os.path.join(ROOT, 'configs', 'default.yaml'))
    cfg['paths'] = {
        'model_dir': os.path.join(ROOT, 'models'),
        'model_file': os.path.join(ROOT, 'models', 'model.pkl'),
        'scaler_file': os.path.join(ROOT, 'models', 'scaler.pkl'),
    }
    train(cfg)
    print('[3] dumped model + scaler to models/')

    # 4. predict helper on a single sample
    predict_mod._MODEL = predict_mod._SCALER = predict_mod._COLUMNS = None
    sample = {
        'CRIM': 0.1, 'ZN': 0.0, 'INDUS': 5.0, 'CHAS': 0,
        'NOX': 0.5, 'RM': 6.0, 'AGE': 50.0, 'DIS': 4.0,
        'RAD': 1, 'TAX': 296, 'PTRATIO': 15.3,
        'B': 396.9, 'LSTAT': 5.0,
    }
    pred = predict_mod.predict_one(sample, model_dir=os.path.join(ROOT, 'models'))
    print('[4] predict_one on sample -> MEDV=%.2f ($1000s)' % pred)
    assert 0.0 < pred < 80.0, 'prediction out of sane range'

    # 5. Flask serving path via the test client
    client = app_module.app.test_client()
    health = client.get('/health')
    assert health.status_code == 200 and health.get_json()['status'] == 'ok'
    resp = client.post('/predict', data=json.dumps(sample),
                       content_type='application/json')
    assert resp.status_code == 200, 'predict route failed: %s' % resp.status_code
    body = resp.get_json()
    assert 'medv' in body
    print('[5] flask /health -> ok ; POST /predict -> %s' % json.dumps(body))

    print('SMOKE OK')


if __name__ == '__main__':
    main()
