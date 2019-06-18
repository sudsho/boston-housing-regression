"""Compare LinearRegression, Ridge, Lasso, RandomForestRegressor on the same split."""
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.preprocess import build_features, fit_scaler, transform
from src.model import build_model
from src.evaluate import compare, print_table


MODEL_CFGS = [
    ('linear', {'type': 'linear'}),
    ('ridge',  {'type': 'ridge', 'ridge_alpha': 1.0}),
    ('lasso',  {'type': 'lasso', 'lasso_alpha': 0.01}),
    ('rf',     {'type': 'rf', 'rf_n_estimators': 400, 'rf_max_depth': 12}),
]


def run(seed=42, test_size=0.2):
    X, y = load_data()
    X_fe = build_features(X, poly_degree=2)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fe, y, test_size=test_size, random_state=seed
    )

    sc = fit_scaler(X_tr)
    X_tr_s = transform(X_tr, sc)
    X_te_s = transform(X_te, sc)

    models = {}
    for name, cfg in MODEL_CFGS:
        m = build_model(cfg)
        m.fit(X_tr_s, y_tr)
        models[name] = m

    rows = compare(models, X_te_s, y_te)
    print_table(rows)
    return rows


if __name__ == '__main__':
    run()
