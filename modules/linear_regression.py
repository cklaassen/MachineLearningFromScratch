from multipledispatch import dispatch
from pandas import DataFrame
from numpy import ndarray


class LinearRegression:
    def __init__(self, X: ndarray, y: ndarray):
        print(X)
        print(y)

    @classmethod
    def from_dataframe(cls, df: DataFrame):
        y = df[df.columns[len(df.columns) - 1]].to_numpy()
        df = df.drop(df.columns[len(df.columns) - 1], axis=1)
        X = df.to_numpy()
        print(y)
        print(X)
        cls(X, y)

    @classmethod
    def from_dataframe_specific_target(cls, df: DataFrame, target: str):
        y = df[df.columns[target]].to_numpy()
        df = df.drop(df.columns[target], axis=1)
        X = df.to_numpy()
        print(y)
        print(X)
        cls(X, y)
