import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.analysis import feature_relationships

df = pd.read_csv("resources/housing.csv")

feature_relationships(df, "median_house_value")
