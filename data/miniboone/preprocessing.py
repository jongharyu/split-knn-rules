"""
preprocessing
"""

from typing import Tuple

import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit


def train_test_stratified_split(df: pd.DataFrame,
                                target: str,
                                test_size: float = 0.2,
                                random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified sampling of the DataFrame into training and test sets

    Args:
        df: DataFrame that is to be split
        target: name of the target column
        test_size: test size, default is 0.2
        random_state: random seed, default is 0

    Returns:
        Tuple of training and test DataFrames
    """

    if target not in df.columns:
        raise IndexError("Column '{0}' is not in the DataFrame!".format(target))

    split = StratifiedShuffleSplit(n_splits=2,
                                   test_size=test_size,
                                   random_state=random_state)

    for train_index, test_index in split.split(df, df[target]):
        # print("Training set size: {0:,}\nTest set size: {1:,}".format(train_index.size, test_index.size))
        return df.loc[train_index], df.loc[test_index]


def x_y_split(df: pd.DataFrame,
              y_col: str) -> Tuple[pd.DataFrame, np.array]:
    """
    Split a DataFrame into x (explanatory variables) and y (target variable)

    Args:
        df: DataFrame that is to be split
        y_col: name of the y column

    Returns:
        Tuple of x (DataFrame) and y (array)
    """

    if y_col not in df.columns:
        raise IndexError("Column '{0}' is not in the DataFrame!".format(y_col))

    x = df.drop(y_col, axis=1).copy()
    y = df[y_col].copy()

    print("Class ratio: {0:.5f}".format(y.sum() / x.shape[0]))

    return x, y


def remove_outliers_knn(x: pd.DataFrame, y: np.array, contamination: float = 0.1) -> Tuple[pd.DataFrame, np.array]:
    """Remove outliers from the training/test set using PyOD's KNN classifier

    Args:
        x: DataFrame containing the X's
        y: target array
        contamination: the amount of contamination of the data set

    Returns:
        x and y with outliers removed
    """
    clf = KNN(contamination=contamination, n_jobs=-1)

    clf.fit(x)

    labels = clf.labels_

    print("{0:.2%} among {1:,} sample points are identified and removed as outliers"
          .format(sum(labels) / x.shape[0], x.shape[0]))

    x = x.iloc[labels == 0]
    y = y[labels == 0]

    return x, y


# From https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class ColumnExcluder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[set(X.columns) - set(self.columns)]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
