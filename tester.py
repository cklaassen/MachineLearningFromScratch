import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.linear_regression import LinearRegression


df = pd.read_csv("resources/housing.csv")

LinearRegression.from_dataframe(df)


