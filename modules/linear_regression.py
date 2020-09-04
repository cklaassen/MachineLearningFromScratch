from multipledispatch import dispatch
from pandas import DataFrame
from numpy import ndarray, random, insert, sum, zeros, power, isnan, delete


class LinearRegression:
    def __init__(self, X: ndarray, y: ndarray):
        self.X = insert(X, 0, 1, axis=1)
        self.y = y
        storage = ~isnan(self.X).any(axis=1)
        self.X = self.X[storage]
        self.y = self.y[storage]
        self.theta = random.random(self.X.shape[1])
        self.values = sum(self.X * self.theta, axis=1)
        print(self.values)
        self.error = sum(power(self.values - self.y, 2))
        print(self.error)

    @classmethod
    def from_dataframe(cls, df: DataFrame):
        y = df[df.columns[len(df.columns) - 1]].to_numpy()
        df = df.drop(df.columns[len(df.columns) - 1], axis=1)
        X = df.to_numpy()
        cls(X, y)

    @classmethod
    def from_dataframe_specific_target(cls, df: DataFrame, target: str):
        y = df[df.columns[target]].to_numpy()
        df = df.drop(df.columns[target], axis=1)
        X = df.to_numpy()
        cls(X, y)

    @dispatch(ndarray, ndarray)
    def fit(self, X, y):
        self.X = X
        self.y = y

    @dispatch(DataFrame)
    def fit(self, df):
        self.y = df[df.columns[len(df.columns) - 1]].to_numpy()
        df = df.drop(df.columns[len(df.columns) - 1], axis=1)
        self.X = df.to_numpy()

    @dispatch(DataFrame, str)
    def fit(self, df, target):
        self.y = df[df.columns[target]].to_numpy()
        df = df.drop(df.columns[target], axis=1)
        self.X = df.to_numpy()

    def _calculate_result(self):
        self.values = sum(self.X * self.theta, axis=1)

    def _calculate_error(self):
        self.error = sum(power(self.values - self.y, 2))

    def _batched_gradient_descent(self, theta, alpha, num_iterations, h, X, y, n):
        cost = zeros(num_iterations)


