import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.linear_regression import LinearRegression


df = pd.read_csv("resources/housing.csv")

n = 300
x = np.arange(-n/2, n/2, 1, dtype=np.float64)

m = np.random.uniform(0.2, 0.5, (n,))
b = np.random.uniform(0, 30, (n,))

y = x*m + b

x = x.reshape((len(x), 1))

# X = np.stack((x, z), axis=-1)
#
lg = LinearRegression()
lg.fit(x, y)
lg._gradient_descent(.001, 10)

y_pred = lg.theta[0] + lg.theta[1] * x
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()
