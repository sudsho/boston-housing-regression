"""5-fold cross validation utility."""
import numpy as np
from sklearn.model_selection import KFold

from src.data import load_data
from src.preprocess import build_features, fit_scaler, transform
from src.model import build_model
from src.evaluate import metrics


def cv_run(model_cfg, n_splits=5, seed=42):
    X, y = load_data()
    X_fe = build_features(X, poly_degree=2)
    X_arr = X_fe.values
    y_arr = y.values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_arr), start=1):
        X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]

        sc = fit_scaler(X_tr)
        X_tr_s = transform(X_tr, sc)
        X_te_s = transform(X_te, sc)

        m = build_model(model_cfg)
        m.fit(X_tr_s, y_tr)
        preds = m.predict(X_te_s)
        d = metrics(y_te, preds)
        d['fold'] = fold
        fold_metrics.append(d)
        print('fold', fold, 'MAE=%.3f RMSE=%.3f R2=%.3f' % (d['mae'], d['rmse'], d['r2']))

    mae_mean = np.mean([f['mae'] for f in fold_metrics])
    rmse_mean = np.mean([f['rmse'] for f in fold_metrics])
    print('mean MAE=%.3f RMSE=%.3f' % (mae_mean, rmse_mean))
    return fold_metrics


if __name__ == '__main__':
    cv_run({'type': 'rf', 'rf_n_estimators': 400, 'rf_max_depth': 12})
