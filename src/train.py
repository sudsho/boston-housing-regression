"""Train a housing model and dump it to disk."""
import os
import logging

import joblib
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import load_data
from src.preprocess import build_features, fit_scaler, transform
from src.model import build_model


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)


def load_config(path='configs/default.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    seed = cfg.get('random_seed', 42)
    test_size = cfg.get('test_size', 0.2)

    X, y = load_data()
    X_fe = build_features(X, poly_degree=cfg['features'].get('poly_degree', 2))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fe, y, test_size=test_size, random_state=seed
    )

    scaler = fit_scaler(X_tr)
    X_tr_s = transform(X_tr, scaler)
    X_te_s = transform(X_te, scaler)

    model = build_model(cfg['model'])
    log.info('training model: %s', cfg['model'].get('type'))
    model.fit(X_tr_s, y_tr)

    preds = model.predict(X_te_s)
    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    log.info('test MAE = %.3f, RMSE = %.3f', mae, rmse)

    out_dir = cfg['paths']['model_dir']
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    joblib.dump(model, cfg['paths']['model_file'])
    joblib.dump(scaler, cfg['paths']['scaler_file'])
    # also dump feature columns so the API knows what to expect
    joblib.dump(list(X_fe.columns), os.path.join(out_dir, 'feature_columns.pkl'))
    log.info('saved model + scaler to %s', out_dir)

    return mae, rmse


if __name__ == '__main__':
    cfg = load_config()
    train(cfg)
