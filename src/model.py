"""Model factory for the housing regression project."""
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


def build_model(cfg):
    """Return an unfitted sklearn estimator based on cfg dict."""
    mtype = cfg.get('type', 'ridge')

    if mtype == 'linear':
        return LinearRegression()

    if mtype == 'ridge':
        alpha = cfg.get('ridge_alpha', 1.0)
        return Ridge(alpha=alpha)

    if mtype == 'lasso':
        alpha = cfg.get('lasso_alpha', 0.01)
        return Lasso(alpha=alpha, max_iter=10000)

    if mtype == 'rf':
        n = cfg.get('rf_n_estimators', 200)
        d = cfg.get('rf_max_depth', 10)
        return RandomForestRegressor(
            n_estimators=n,
            max_depth=d,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError('unknown model type: ' + str(mtype))
