from multipledispatch import dispatch
from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy import ndarray, random, insert, sum, zeros, power, isnan, delete, square


class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        self.values = None
        self.theta = None
        self.cost = []

    @dispatch(ndarray, ndarray)
    def fit(self, X, y):
        self.X = X
        self.y = y
        self._data_cleaning()
        self._initiate_data()

    @dispatch(DataFrame)
    def fit(self, df):
        self.y = df[df.columns[len(df.columns) - 1]].to_numpy()
        df = df.drop(df.columns[len(df.columns) - 1], axis=1)
        self.X = df.to_numpy()
        self._data_cleaning()
        self._initiate_data()

    @dispatch(DataFrame, str)
    def fit(self, df, target):
        self.y = df[df.columns[target]].to_numpy()
        df = df.drop(df.columns[target], axis=1)
        self.X = df.to_numpy()
        self._data_cleaning()
        self._initiate_data()

    def _data_cleaning(self):
        # Insert 1 in front of every row of data for constant in equation
        self.X = insert(self.X, 0, 1, axis=1)


        # Check for NaN in both X and y data, remove any that exist
        storage = ~isnan(self.X).any(axis=1)
        self.X = self.X[storage]
        self.y = self.y[storage]
        # storage = ~isnan(self.y).any(axis=1)
        # self.X = self.X[storage]
        # self.y = self.y[storage]

    def _initiate_data(self):
        self.theta = random.random(self.X.shape[1])

        self.values = sum(self.X * self.theta, axis=1)
        self.error = sum(power(self.values - self.y, 2))

    def _calculate_result(self):
        self.values = sum(self.X * self.theta, axis=1)

    def _calculate_error(self):
        self.error = sum(power(self.values - self.y, 2))


    def _gradient_descent(self, alpha, num_iterations):
        for i in range(num_iterations):
            self.theta = self.theta - (alpha / self.X.shape[0]) * sum((self.values - self.y) * self.X.transpose())
            if i % 10 == 0:
                print(self.theta)
                print(self.X[0])
                print(self.values[0])
            self._calculate_result()
            self.cost.append((1 / self.X.shape[0]) * 0.5 * sum(square(self.values - self.y)))
        plt.plot(self.cost)
        plt.show()


