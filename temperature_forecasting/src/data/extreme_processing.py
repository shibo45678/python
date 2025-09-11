# import pandas as pd
# import numpy as np
# from scipy import stats
# from ..data.processing import DataPreprocessor
#
#
#
# class ExtremeDataHandler(DataPreprocessor):
#     """极端数据专用处理"""
#     def __init__(self,df:pd.DataFrame):
#         super().__init__() # 继承父类构造函数
#         self.extreme_flags={} # 存储极端值标志

      def custom_handler():
         """业务判断：物理上不可能的值 / 极端数据
            返回表格，根据索引处理"""
#
#     def single_remove_zscore(self,
#                                target_column:str=None,
#                                threshold: int = 3):  # z = (x - μ) / σ 单位标准差 >=3个标准差算异常
#         """使用Z-score方法移除异常值"""
#         print(f"处理{target_column}列异常值(zscore)...")
#
#         if target_column in self.numeric_columns:
#             z_scores = np.abs(stats.zscore(self.df[target_column]))
#             print(f"列{target_column}的{threshold}倍标准差外的异常值：{self.df[target_column][z_scores >= threshold]}")
#
#             before_count = len(self.df)
#             self.df = self.df[z_scores < threshold]
#             after_count = len(self.df)
#             if before_count != after_count:
#                 print(f"列'{target_column}':移除了 {before_count - after_count} 个异常值")
#         self.history.append("处理[单列]异常值(zscore)")
#         return self
#
#     def single_remove_iqr(self, target_column=None,threshold: float = 1.5):# 不支持多特征联合异常处理
#         """使用IQR方法移除异常值"""
#         print(f"处理{target_column}列异常值(iqr)...")
#         if target_column in self.numeric_columns:
#             Q1, Q3 = self.df[target_column].quantile([0.25, 0.75])
#             IQR = Q3 - Q1
#             lower_bound = Q1 - threshold * IQR
#             upper_bound = Q3 + threshold * IQR
#             print(f"列{target_column}异常低值：{self.df[target_column][self.df[target_column] < lower_bound]}")
#             print(f"列{target_column}异常高值：{self.df[target_column][self.df[target_column] > upper_bound]}")
#
#             before_count = len(self.df)
#             self.df = self.df[self.df[target_column] >= lower_bound and self.df[target_column] <= upper_bound]
#             after_count = len(self.df)
#             if before_count != after_count:
#                 print(f"列'{target_column}':移除了 {before_count - after_count} 个异常值")
#
#         self.history.append("处理[单列]异常值(iqr)")
#         return self
#
#     def multivariate_remove_outliers(self, contamination=0.025):  # 预期异常比例 ≈2.5%
#         """多变量联合异常检测
#         多变量联合分析，不是逐列处理
#         某个点可能单个特征正常，但多个特征的组合异常
#         """
#         from sklearn.ensemble import IsolationForest
#
#         # 1.使用隔离森林检测整体异常
#         iso_forest = IsolationForest(
#             contamination=contamination,
#             random_state=42
#         )
#         outliers = iso_forest.fit_predict(self.df[self.numeric_columns]) # 注意是否有删除
#         # 2.标记异常点
#         self.df['is_outlier'] = outliers == -1
#         print(f"检测到{self.df['is_outlier'].sum()}个多变量异常点")
#
#         return self
#
#
#
#
#
#
#
#     def analyze_extreme_patterns():
#         pass
#     def assess_extreme_risks():
#         pass
#



