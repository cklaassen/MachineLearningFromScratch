import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.linear_regression import _init
from modules.LinearRegression import LinearRegression



df = pd.read_csv("resources/housing.csv")

y = df["median_house_value"].to_numpy()
df = df.drop("median_house_value", axis=1)
X = df.to_numpy()

_init(df)


