import numpy as np
import pandas as pd
from multipledispatch import dispatch

class LogisticRegression:
    def __init__(self, X):
        self.X = X
        self.theta = None
        self.y = None
        self.values = None
        self.cost = None

    @dispatch(np.ndarray, np.ndarray)
    def fit(self, X, y):
        self.X = X
        self.y = y

    @staticmethod
    def _sigmoid(X):
        return 1/(1 + np.exp(-X))

    @dispatch(np.ndarray, np.ndarray)
    def fit(self, X, y):
        self.X = X
        self.y = y
        self._data_cleaning()
        self._initiate_data()
        self._gradient_descent()

    @dispatch(pd.DataFrame)
    def fit(self, df):
        self.y = df[df.columns[len(df.columns) - 1]].to_numpy()
        df = df.drop(df.columns[len(df.columns) - 1], axis=1)
        self.X = df.to_numpy()
        self._data_cleaning()
        self._initiate_data()
        self._gradient_descent()

    @dispatch(pd.DataFrame, str)
    def fit(self, df, target):
        self.y = df[df.columns[target]].to_numpy()
        df = df.drop(df.columns[target], axis=1)
        self.X = df.to_numpy()
        self._data_cleaning()
        self._initiate_data()
        self._gradient_descent()

    def _data_cleaning(self):
        # Insert 1 in front of every row of data for constant in equation
        self.X = np.insert(self.X, 0, 1, axis=1)

        # Check for NaN in both X and y data, remove any that exist
        storage = ~np.isnan(self.X).any(axis=1)
        self.X = self.X[storage]
        self.y = self.y[storage]
        # storage = ~isnan(self.y).any(axis=1)
        # self.X = self.X[storage]
        # self.y = self.y[storage]

    def _initiate_data(self):
        self.theta = np.random.random(self.X.shape[1])

        self.values = sum(self.X * self.theta, axis=1)
        self.error = sum(np.power(self.values - self.y, 2))

    def _calculate(self):
        self.values = self._sigmoid(self.theta * self.X)
