from numpy import sum
from modules.regression import Regression


class LinearRegressionMine(Regression):
    def predict(self, X):
        return sum(X * self.theta, axis=1)
