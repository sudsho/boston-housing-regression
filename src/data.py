"""Load the Boston housing dataset."""
import pandas as pd
from sklearn.datasets import load_boston


FEATURE_NAMES = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]
TARGET_NAME = 'MEDV'


def load_data():
    """Returns features dataframe X and target series y."""
    raw = load_boston()
    X = pd.DataFrame(raw.data, columns=FEATURE_NAMES)
    y = pd.Series(raw.target, name=TARGET_NAME)
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print(X.head())
