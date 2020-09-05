import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.linear_regression import LinearRegressionMine
from sklearn.linear_model import LinearRegression
from modules.logistic_regression import LogisticRegression


x = np.arange(-300/2, 300/2, 1, dtype=np.float64)

y1 = [0] * 150
y2 = [1] * 150

y = np.array((y1, y2))
y =y.reshape(300)
x = x.reshape((300, 1))


# df = pd.read_csv("resources/housing.csv")
#
# n = 300
# x = np.arange(-n/2, n/2, 1, dtype=np.float64)
#
# m = np.random.uniform(-0.5, -.1, (n,))
# b = np.random.uniform(-10, 30, (n,))
#
# y = x*m + b
#
# x = x.reshape((len(x), 1))
#
# # X = np.stack((x, z), axis=-1)
#
# reg = LinearRegression().fit(x, y)
# pred = reg.predict(x)
#
#
# lg = LinearRegressionMine()
# lg.fit(x, y)
# print(lg.predict(x))
# print(lg.performance())
#
# y_pred = lg.theta[0] + lg.theta[1] * x
#
# print(f'{lg.theta[1]}x + {lg.theta[0]}')
#
# plt.scatter(x, y)
# plt.plot(x, pred, 'y*')
# plt.plot(x, y_pred, 'g-')
# plt.show()

log_R = LogisticRegression()

log_R.fit(x, y)

plt.scatter(x, y)
plt.plot(x, log_R.values)
plt.show()
