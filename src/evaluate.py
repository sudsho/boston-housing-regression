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


def compare(models, X_test, y_test):
    """Run a dict of {name: fitted_model} on the test data, return a list of rows."""
    rows = []
    for name, m in models.items():
        preds = m.predict(X_test)
        d = metrics(y_test, preds)
        d['model'] = name
        rows.append(d)
    return rows


def print_table(rows):
    print('model'.ljust(14), 'MAE'.rjust(8), 'RMSE'.rjust(8), 'R2'.rjust(8))
    for r in rows:
        print(
            r['model'].ljust(14),
            ('%.3f' % r['mae']).rjust(8),
            ('%.3f' % r['rmse']).rjust(8),
            ('%.3f' % r['r2']).rjust(8),
        )
