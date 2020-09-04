import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.linear_regression import LinearRegression


df = pd.read_csv("resources/housing.csv")

x = np.linspace(-5,5,100)
y = 3 * x + 1
z = 2 * y + 3 * x + 5

X = np.stack((x, z), axis=-1)

x = x.reshape((len(x), 1))
print(x)

lg = LinearRegression()
lg.fit(x, y)
lg._gradient_descent(.02, 100)
