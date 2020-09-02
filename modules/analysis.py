from multipledispatch import dispatch
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import math


@dispatch(DataFrame, str)
def feature_relationships(df, column_target):
    fig = plt.figure(figsize=(14, (len(df.columns) / 2) * 3))
    fig.suptitle('Relationships of Data')
    target = df[column_target]
    df = df.drop(column_target, axis=1)
    for i, col in enumerate(df.columns):
        plt.subplot(math.ceil(len(df.columns) / 2), 2, i+1)
        plt.plot(df[col], target, marker='.', linestyle='none')
        plt.title(col + " vs " + column_target)
        plt.tight_layout()
    plt.show()


@dispatch(DataFrame)
def feature_relationships(df):
    fig = plt.figure(figsize=(14, ((len(df.columns) - 1) / 2) * 3))
    fig.suptitle('Relationships of Data')
    column_target = df.columns[len(df.columns) - 1]
    target = df[column_target]
    df = df.drop(column_target, axis=1)
    for i, col in enumerate(df.columns):
        plt.subplot(math.ceil(len(df.columns) / 2), 2, i+1)
        plt.plot(df[col], target, marker='.', linestyle='none')
        plt.title(col + " vs " + column_target)
        plt.tight_layout()
    plt.show()


def heatmap(df):
    fig = plt.subplots(figsize=(10, 10))
    sns.set(font_scale=1.5)
    sns.heatmap(df.corr(), square=True, cbar=True, annot=True, annot_kws={'size': 10})
    plt.show()
