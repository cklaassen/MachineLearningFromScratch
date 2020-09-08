from multipledispatch import dispatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Regression(object):
    def __init__(self):
        self.X = None
        self.theta = None
        self.y = None
        self.values = None
        self.cost = []

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

        self.values = np.sum(self.X * self.theta, axis=1)
        self.error = sum(np.power(self.values - self.y, 2))

    def _run(self, alpha):
        self.theta = self.theta - ((alpha / self.X.shape[0]) * np.sum((self.values - self.y) * self.X.transpose()))
        self.values = np.sum(self.X * self.theta, axis=1)
        self.cost.append((1 / self.X.shape[0]) * 0.5 * sum(np.square(self.values - self.y)))

    def _gradient_descent(self, alpha=.01, num_iterations=300):
        if num_iterations is not None:
            for i in range(num_iterations):
                self._run(alpha)
        else:
            for i in range(30):
                self._run(alpha)
            while self.cost[len(self.cost) - 1] < self.cost[len(self.cost) - 2]:
                self._run(alpha)
        print(self.cost)
        plt.plot(self.cost)
        plt.show()

    def _calculate_error(self):
        self.error = sum(np.power(self.values - self.y, 2))

    def performance(self):
        self._calculate_error()
        return self.error
