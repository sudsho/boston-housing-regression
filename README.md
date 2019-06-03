# boston-housing-regression

Predicting median house values in Boston suburbs from 13 features. Uses the
classic Boston housing dataset shipped with scikit-learn.

## Problem

Given 13 attributes about a town in the Boston metropolitan area (per-capita
crime rate, average rooms per dwelling, % lower status of the population,
etc.), predict the median value of owner occupied homes (`MEDV`, in $1000s).

This is a small (506 rows) but classic regression benchmark. Good for
practicing feature engineering, comparing linear vs tree models, and putting a
small Flask API in front of a saved sklearn model.

## Setup

```
pip install -r requirements.txt
```

## Status

scaffolding the project.
