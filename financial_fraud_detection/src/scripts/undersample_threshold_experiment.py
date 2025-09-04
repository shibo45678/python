import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union
from ..models.logistic_regression import LogisticRegressionModel
from ..utils.config import get_config
from ..evaluation.visualization import plot_confusion_matrix_with_metrics


class UndersampleExperiment:
    def __init__(self):
        """从全局配置获取参数"""
        self.config = get_config()
        print(f"下采样阈值测试中的最优参数C取值: {self.config.best_c}")

        self.lr = LogisticRegressionModel(C=self.config.best_c)
        self.lr.fit(self.config.undersample_data[0],
                    self.config.undersample_data[2].values.ravel())  # 用下采样数据训练模型，按照train_test_split的结果顺序取

    def test_best_thres(self):
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # 样本归属各类01的概率值 a sample belongs to each class. a 2-dimensional NumPy array  shape(n_samples, n_classes) n_classes二分类=2
        # 结果每个'样本' [0.89  0.11] 归属类的概率
        y_pred_undersample_proba = self.lr.predict_proba(self.config.undersample_data[1])

        # 创建一个大图和3*3的子图网格
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()  # 将2D轴数组展平为1D，便于迭代

        for i, threshold in enumerate(thresholds):
            # 布尔过滤 : 归属为异常值的概率大于该阈值 的数据.
            # 假设i=0.1 ,代表只要样本的异常值概率大于0.1，就会被筛选出标记为 True，true=1
            # 混淆矩阵时，异常判断过于宽松，很多被算为异常值。recall=1 也就是代表，所有都被算作异常值。
            y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > threshold

            # 在当前子图上绘制
            plot_confusion_matrix_with_metrics(self.config.undersample_data[3],
                                               y_test_predictions_high_recall,
                                               ax=axes[i],  # 传递子图对象
                                               title=f'Threshold >= {threshold}')
        plt.tight_layout()
        plt.show()

    def run_best_thres(self,
                       thres: float = None,
                       data_train: pd.DataFrame = None,
                       data_true: Union[pd.DataFrame, pd.Series, np.ndarray] = None):
        y_pred_undersample_proba = self.lr.predict_proba(data_train)
        y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > thres

        plot_confusion_matrix_with_metrics(data_true,
                                           y_test_predictions_high_recall,
                                           title='Threshold >= %s' % thres)
