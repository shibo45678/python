#  评估指标
from typing import Union, Optional, Dict
import pandas as pd
import numpy as np
from ..evaluation.visualization import plot_confusion_matrix_with_metrics


def evaluate_model_performance(
        y_true: Union[np.ndarray, pd.Series, pd.DataFrame],
        y_pred: Union[np.ndarray, pd.Series, pd.DataFrame],
        model_name: Optional[str] = None,
        report:bool=False,
        plot:bool= False) -> Dict: # 可选画图

    from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
    # 混淆矩阵、召回率（评估二分类回归的标准）
    result = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

    # 分类报告开关
    if report:
        from sklearn.metrics import classification_report

        print("=== 模型性能评估 ===")
        if model_name:
            print(f"模型: {model_name}")
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"精确率: {result['precision']:.4f}")
        print(f"召回率: {result['recall']:.4f}")
        print(f"F1分数: {result['f1']:.4f}")

        print("\n分类报告:")
        print(classification_report(y_true, y_pred))

    # 绘图开关
    if plot :
        plot_confusion_matrix_with_metrics(y_true, y_pred,save=False)

    return result
