# 结果可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Union


# 计算混淆矩阵（相同测试集上的真实值、预测值） a 1D list, NumPy array, or pandas Series. [0, 1, 1, 0, 1]
def plot_confusion_matrix_with_metrics(y_true: Union[np.ndarray, pd.Series],
                                       y_pred: Union[np.ndarray, pd.Series],
                                       title: str = 'Confusion Matrix',
                                       save: bool=False
                                       ) -> None:
    """
    绘制混淆矩阵并计算相关指标
    Parameters:
    y_ture: 真实标签
    y_pred: 预测标签
    title: 图表标题
    """
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    # 有了混淆矩阵的情况下，计算TP/(FN+TP) 。FN 真实值=1，预测值=0 先y后x
    recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])
    print(f"召回率: {recall:.4f}")

    '''
    绘制混淆矩阵（cm:计算出的混淆矩阵的值，classes：分类标签，cmap:绘图样式)
    '''

    # 绘制混淆矩阵
    classes = ['0', '1']
    cm = cnf_matrix
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()  # 灰度条
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    import itertools
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()
    if save :
        plt.savefig(f"{title}.png")
    plt.close()

    return None
