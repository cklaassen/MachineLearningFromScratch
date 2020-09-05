import numpy as np
import pandas as pd
from multipledispatch import dispatch
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self):
        self.X = None
        self.theta = None
        self.y = None
        self.values = None
        self.cost = []

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

        self.values = self._sigmoid(np.sum(self.X * self.theta, axis=1))

    def _calculate(self):
        self.values = self._sigmoid(np.sum(self.X * self.theta, axis=1))

    def _run(self, alpha):
        self.theta = self.theta - ((alpha / self.X.shape[0]) * np.sum((self.values - self.y) * self.X.transpose()))
        self._calculate()
        self.cost.append((1 / self.X.shape[0]) * 0.5 * sum((self.y * np.log(self.values)) + ((1 - self.y) * np.log(1 - self.values))))

    def _gradient_descent(self, alpha=.000001, num_iterations=None):
        if num_iterations is not None:
            for i in range(num_iterations):
                self._run(alpha)
        else:
            for i in range(30):
                self._run(alpha)
            while self.cost[len(self.cost) - 1] < self.cost[len(self.cost) - 2]:
                self._run(alpha)
        plt.plot(self.cost)
        plt.show()

