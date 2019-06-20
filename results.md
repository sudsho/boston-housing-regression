# Results

Held-out test set is 20% of 506 rows = 102 rows. Random seed 42.

| Model              | MAE  | RMSE | R2   |
|--------------------|------|------|------|
| LinearRegression   | 3.31 | 4.69 | 0.74 |
| Ridge (alpha=1)    | 3.27 | 4.62 | 0.75 |
| Lasso (alpha=0.01) | 3.30 | 4.66 | 0.74 |
| RandomForest (400) | 2.04 | 2.94 | 0.89 |

Numbers come from `python -m src.benchmark` after running with the engineered
features (log of CRIM/LSTAT/DIS, squared RM and LSTAT, RM x LSTAT interaction,
B dropped, StandardScaler applied).

The RandomForest wins comfortably on this dataset. The linear models are
nearly tied; Ridge is marginally best.
