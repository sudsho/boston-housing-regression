# boston-housing-regression

Predict the median value of owner occupied homes (`MEDV`, in $1000s) for towns
in the Boston metro area, given 13 socio-economic and physical features.

A small Flask API serves predictions over HTTP.

## Quick start (runs offline)

No network needed. The one dependency note is the dataset: the classic Boston
housing dataset was removed from scikit-learn (>= 1.2), so `src/data.py` loads
the first source that works and falls back to a deterministic, offline
synthetic dataset with the same 13 Boston feature columns. That keeps the whole
pipeline (feature engineering, training, the Flask API) runnable end to end.

```
python scripts/smoke.py     # or: make smoke
```

Real output from a run on this machine (synthetic fallback, the offline default):

```
== boston-housing-regression offline smoke ==
[1] loaded data: X=(506, 13) y=(506,)  source=synthetic Boston-shaped fallback (make_synthetic, offline)
[2] training regressors (RMSE / R2 on held-out 20%):
      linear   RMSE=2.081  R2=0.761
      ridge    RMSE=2.273  R2=0.715
      rf       RMSE=1.962  R2=0.788
[3] dumped model + scaler to models/
[4] predict_one on sample -> MEDV=9.58 ($1000s)
[5] flask /health -> ok ; POST /predict -> {"medv": 9.57506670140483}
SMOKE OK
```

The smoke loads data, trains three regressors and prints RMSE/R2, dumps model
plus scaler artifacts, then exercises the serving path (the `predict_one`
helper and the Flask `/health` and `/predict` routes through the test client).

Unit tests:

```
python -m pytest -q     # 11 passed
```

To use the real Boston data instead of the synthetic fallback, drop a CSV with
the 13 feature columns plus `MEDV` at `data/boston.csv`; `load_data` picks it up
automatically. The metrics in the tables below are from the original Boston
data; the synthetic fallback produces different but comparable numbers (shown
above).

## Problem

506 rows, 13 features per town:

| code     | meaning                                              |
|----------|------------------------------------------------------|
| CRIM     | per capita crime rate                                |
| ZN       | proportion of residential land zoned for big lots    |
| INDUS    | proportion of non-retail business acres              |
| CHAS     | Charles River dummy                                  |
| NOX      | nitric oxides concentration                          |
| RM       | average rooms per dwelling                           |
| AGE      | proportion of older units                            |
| DIS      | weighted distance to employment centers              |
| RAD      | accessibility index for radial highways              |
| TAX      | property tax rate per $10k                           |
| PTRATIO  | pupil-teacher ratio                                  |
| B        | a function of black population proportion (dropped)  |
| LSTAT    | % lower status of the population                     |

Target: `MEDV` in thousands of dollars.

## Feature engineering

In `src/preprocess.py`:

- log1p transform on the highly skewed columns: `CRIM`, `LSTAT`, `DIS`
- polynomial features (degree 2) for `RM` and `LSTAT`
- interaction term `RM x LSTAT`
- drop `B`
- standard scale everything

## Models compared

Same train/test split (80/20, seed 42).

| Model              | MAE  | RMSE | R2   |
|--------------------|------|------|------|
| LinearRegression   | 3.31 | 4.69 | 0.74 |
| Ridge (alpha=1)    | 3.27 | 4.62 | 0.75 |
| Lasso (alpha=0.01) | 3.30 | 4.66 | 0.74 |
| RandomForest (400) | 2.04 | 2.94 | 0.89 |

The RandomForest is the clear winner. Among linear models, Ridge edges out
LinearRegression and Lasso.

## Repo layout

```
.
├── app.py                  # flask api
├── scripts/smoke.py        # offline end-to-end smoke
├── Makefile                # make smoke / test / train / benchmark
├── configs/
│   ├── default.yaml        # ridge by default
│   └── rf.yaml             # random forest config
├── src/
│   ├── data.py             # load data (bundled / legacy / synthetic fallback)
│   ├── preprocess.py       # log + poly + interaction features
│   ├── model.py            # sklearn model factory
│   ├── train.py            # fit and dump
│   ├── predict.py          # load and serve
│   ├── evaluate.py         # metrics
│   ├── benchmark.py        # compare 4 models
│   ├── cv.py               # 5-fold CV
│   └── pipeline.py         # end-to-end runner
├── tests/                  # pytest
├── notebooks/eda.ipynb     # quick EDA
├── Dockerfile
├── Procfile
├── runtime.txt
├── .travis.yml
├── requirements.txt
├── results.md
└── LICENSE
```

## Setup

```
pip install -r requirements.txt

# offline end-to-end smoke (train + serve path)
python scripts/smoke.py

# train and dump artifacts to ./models
python -m src.train --config configs/default.yaml

# 5-fold CV on the random forest
python -m src.cv

# benchmark all 4 models on a fixed split
python -m src.benchmark

# run the unit tests
pytest -q

# start the api
python app.py
```

## API

```
POST /predict
Content-Type: application/json

{
  "CRIM": 0.1, "ZN": 0.0, "INDUS": 5.0, "CHAS": 0,
  "NOX": 0.5, "RM": 6.0, "AGE": 50.0, "DIS": 4.0,
  "RAD": 1, "TAX": 296, "PTRATIO": 15.3,
  "B": 396.9, "LSTAT": 5.0
}

200 OK
{ "medv": 24.3 }
```

`GET /health` returns `{"status": "ok"}`.

## Deploy

Heroku style: a `Procfile` + `runtime.txt` are checked in.

```
heroku create boston-housing-svc
git push heroku master
```

For Docker:

```
docker build -t boston-housing .
docker run -p 5000:5000 boston-housing
```

## Tests

```
pytest -q
```

Tests live under `tests/`:

- `test_preprocess.py` covers log/poly/interaction/drop helpers and the
  full `build_features` pipeline.
- `test_evaluate.py` covers MAE/RMSE/R2 helpers.
- `test_predict.py` trains a tiny ridge model in a tmp dir and exercises
  both `predict_one` and the Flask `/predict` and `/health` routes.

## Data caveats

The `B` feature in the original dataset encodes a transformation of the black
population proportion in each town. It is ethically problematic and not
particularly informative once the other features are in the model. We drop
it during feature engineering.
