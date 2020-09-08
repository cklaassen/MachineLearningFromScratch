from multipledispatch import dispatch
from modules.regression import Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression(Regression):
    @staticmethod
    def _sigmoid(X):
        return 1/(1 + np.exp(-X))

    def _initiate_data(self):
        self.theta = np.random.random(self.X.shape[1])

        self.values = self._sigmoid(np.sum(self.X * self.theta, axis=1))

    def _calculate(self):
        self.values = self._sigmoid(np.sum(self.X * self.theta, axis=1))

    def _run(self, alpha):
        self.theta = self.theta - alpha * (np.dot(self.X.T, (self.values - self.y)) / self.y.shape[0])
        self._calculate()
        self.cost.append((-self.y * np.log(self.values) - (1 - self.y) * np.log(1 - self.values)).mean())

