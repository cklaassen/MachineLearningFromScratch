import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.linear_regression import LinearRegression

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


df = pd.read_csv("resources/housing.csv")

x = np.linspace(-5,5,100)
y = 3 * x + 1
z = 2 * y + 3 * x + 5

y = y * np.random.random(100)
z = z * np.random.random(100)


X = np.stack((x, z), axis=-1)

lg = LinearRegression()
lg.fit(X, y)
lg._gradient_descent(.001, 10)

y_pred = lg.theta[0] + lg.theta[1] * x + lg.theta[2] * z

# ax.plot(x, z, y_pred)
# ax.scatter(xs=x, ys=z, zs=y)
# plt.show()
