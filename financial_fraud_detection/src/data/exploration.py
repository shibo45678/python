import matplotlib.pyplot as plt
import pandas as pd

'''因变量类别分布分析'''


def plot_class_distribution(y: pd.Series,
                            title: str = 'Fraud class histogram') -> None:
    # 查看分布情况  pd.value_counts:Return a Series containing the frequency of each distinct row in the Dataframe
    count_classes = y.value_counts(sort=True,  # Sort by frequencies
                                   ascending=True,
                                   dropna=True)  # 空值不计数
    print(count_classes)

    # 画图
    count_classes.plot(kind='bar')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()
