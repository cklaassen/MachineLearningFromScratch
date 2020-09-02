import matplotlib.pyplot as plt
import math


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
