"""Evaluate one or more trained models on the test split."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred):
    """Return MAE, RMSE, R2 as a dict."""
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
    }


def pretty(d):
    return 'MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}'.format(**d)
