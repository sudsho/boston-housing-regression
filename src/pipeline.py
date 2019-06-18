"""End-to-end pipeline: load -> features -> split -> scale -> fit -> evaluate.

Useful for benchmarking multiple models on the same split without rewriting
boilerplate. Returns the fitted artifacts so the caller can save them.
"""
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.preprocess import build_features, fit_scaler, transform
from src.model import build_model
from src.evaluate import metrics


def run(cfg):
    seed = cfg.get('random_seed', 42)
    test_size = cfg.get('test_size', 0.2)
    poly_degree = cfg['features'].get('poly_degree', 2)

    X, y = load_data()
    X_fe = build_features(X, poly_degree=poly_degree)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fe, y, test_size=test_size, random_state=seed
    )

    scaler = fit_scaler(X_tr)
    X_tr_s = transform(X_tr, scaler)
    X_te_s = transform(X_te, scaler)

    model = build_model(cfg['model'])
    model.fit(X_tr_s, y_tr)
    preds = model.predict(X_te_s)

    return {
        'model': model,
        'scaler': scaler,
        'columns': list(X_fe.columns),
        'metrics': metrics(y_te, preds),
        'X_test': X_te_s,
        'y_test': y_te,
    }
