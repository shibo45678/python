# import matplotlib.pyplot as plt
# import pandas as pd
#
# class DataExplorer:
#
#
#
# def main():
#     # 探索数据（继承自DataPreprocessor）
#     processor = DataPreprocessor(df)
#     processed_df = processor.clean_data()
#     explorer = DataExplorer(processed_df)
#
# def handle_missing_values(self,
#                           cat_strategy: str = 'custom',  # 支持众数填充/自定义填充
#                           num_strategy: str = 'mean', num_fill_value=None):  # 支持均值/众数/中位数/常数填充(num_fill_value)
#     """处理缺失值"""
#     print("==========统计空值结果==========")
#     print(self.df.isna().sum())