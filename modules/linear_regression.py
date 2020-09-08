from multipledispatch import dispatch
from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy import ndarray, random, insert, sum, power, isnan, square
from modules.regression import Regression


class LinearRegressionMine(Regression):
    def _calculate_result(self):
        self.values = sum(self.X * self.theta, axis=1)

    def _calculate_error(self):
        self.error = sum(power(self.values - self.y, 2))

    def _run(self, alpha):
        self.theta = self.theta - ((alpha / self.X.shape[0]) * sum((self.values - self.y) * self.X.transpose()))
        self._calculate_result()
        self.cost.append((1 / self.X.shape[0]) * 0.5 * sum(square(self.values - self.y)))

    def predict(self, X):
        return sum(X * self.theta, axis=1)

    def performance(self):
        self._calculate_error()
        return self.error
