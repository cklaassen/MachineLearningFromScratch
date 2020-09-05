import numpy as np

class LogisticRegression:
    def __init__(self, X):
        self.X = X

    def _sigmoid(self):
        return 1/(1 + np.exp(-self.X))
