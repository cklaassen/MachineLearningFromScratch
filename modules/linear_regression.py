from multipledispatch import dispatch
from pandas import DataFrame
from numpy import ndarray, random, insert, sum, zeros, power, isnan, delete


class LinearRegression:
    def __init__(self, X: ndarray, y: ndarray):
        self.X = insert(X, 0, 1, axis=1)
        missing_val_iterator = 0
        self.y = y
        i = 0
        nan_list = sum(isnan(self.X), axis=1)
        print(nan_list)
        while i < len(self.X):
            if nan_list[i] > 0:
                missing_val_iterator += 1
                self.X = delete(self.X, i)
                self.y = delete(self.y, i)
            else:
                print(i)
                i += 1
        print(missing_val_iterator)
        missing_val_iterator = 0
        for nan_list in (isnan(self.X)):
            if sum(nan_list) > 0:
                missing_val_iterator += 1
        print(missing_val_iterator)
        num_params = self.X.shape[1]
        self.theta = random.random(num_params)
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

    def _calculate_result(self):
        self.values = sum(self.X * self.theta, axis=1)

    def _calculate_error(self):
        self.error = sum(power(self.values - self.y, 2))

    def _batched_gradient_descent(self, theta, alpha, num_iterations, h, X, y, n):
        cost = zeros(num_iterations)


