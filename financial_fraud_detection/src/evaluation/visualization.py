# 结果可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Union,Optional,Any


# 计算混淆矩阵（相同测试集上的真实值、预测值） a 1D list, NumPy array, or pandas Series. [0, 1, 1, 0, 1]
def plot_confusion_matrix_with_metrics(y_true: Union[np.ndarray, pd.Series],
                                       y_pred: Union[np.ndarray, pd.Series],
                                       ax: plt.Axes = None,  # 接收子图对象 如果为None则创建新图
                                       title: str = 'Confusion Matrix',
                                       save: bool = False
                                       ) -> Optional[Any]:
    #  matplotlib.image.AxesImage: 图像对象（当ax不为None时）
    #  None: 当创建独立图形时
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    # 有了混淆矩阵的情况下，计算TP/(FN+TP) 。FN 真实值=1，预测值=0 先y后x
    recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])
    print(f"召回率: {recall:.4f}")

    '''
    绘制混淆矩阵（cm:计算出的混淆矩阵的值，classes：分类标签，cmap:绘图样式)
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone =True
    else:
        standalone = False

    # 绘制混淆矩阵
    classes = ['0', '1']
    cm = cnf_matrix
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)

    if standalone:
        plt.colorbar(im, ax=ax)  # 灰度条

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    import itertools
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    # 只有在没有提供子图对象时才显示和关闭
    if standalone:
        plt.tight_layout()
        if save:
            plt.savefig(f"{title}.png")
        plt.show()
        plt.close()

    return im
