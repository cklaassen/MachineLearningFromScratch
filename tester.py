import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.linear_regression import LinearRegression


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
