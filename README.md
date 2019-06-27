# boston-housing-regression

Predict the median value of owner occupied homes (`MEDV`, in $1000s) for towns
in the Boston metro area, given 13 socio-economic and physical features. Uses
the classic Boston housing dataset bundled with scikit-learn.

A small Flask API serves predictions over HTTP.

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
├── configs/
│   ├── default.yaml        # ridge by default
│   └── rf.yaml             # random forest config
├── src/
│   ├── data.py             # load Boston dataset
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
python -m src.train --config configs/default.yaml
python -m src.benchmark
pytest -q
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

## Data caveats

The `B` feature in the original dataset encodes a transformation of the black
population proportion in each town. It is ethically problematic and not
particularly informative once the other features are in the model. We drop
it during feature engineering.
