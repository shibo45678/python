import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from ..utils.config import get_config

'''逻辑斯蒂回归分类-信用卡欺诈检测(01变量/样本不均衡）
商业价值：
1. 金融领域：对互联网金融产品进行用户分析，同时可以监测商业欺诈
2. 电商领域：对客户进行精确分类，从而实现精准营销
3. 医疗领域：对疾病诊断结果进行快速检验
'''

class LogisticRegressionModel:
    """封装逻辑回归模型"""

    def __init__(self, C=1.0):
        config = get_config()
        self.model = LogisticRegression(C=C,**config.model_config) # 使用 ** 解包字典为关键字参数
        self.is_fitted = False # 训练后更新状态 initialize the fitted state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.is_fitted = True  # It is purely about the state or history of the model itself.

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:  # 模型学习阶段结束，内部已有参数设置，可进行预测
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def get_params(self) -> dict:
        """获取模型参数"""
        return self.model.get_params()
