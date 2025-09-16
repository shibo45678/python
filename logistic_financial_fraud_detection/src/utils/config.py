from typing import Tuple,Dict
import pandas as pd

# 全局配置存储
class GlobalConfig:
    def __init__(self):
        self.best_c = None  # 下采样之前参数调整C会变换，获取下采样后的最优 C
        self.undersample_data = None # 下采样数据
        self.model_config = {'penalty': 'l1',
                             'random_state': 42,
                             'solver': 'liblinear',
                             'max_iter': 12000,
                             'tol': 1e-3}   # 放宽收敛标准
# 默认配置先放宽容忍度。看结果'收敛'和'性能'，确定最优C（收敛且召回率稳定最佳）。
# 那么就可以适当调整其他参数，main

# 创建全局配置实例
config = GlobalConfig()

def set_config(best_c:float,
               undersample_data:Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame],
               model_config:Dict):

    config.best_c = best_c
    config.undersample_data = undersample_data
    config.model_config = model_config


def get_config():

    return config