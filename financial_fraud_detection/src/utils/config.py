from typing import Tuple,Dict
import pandas as pd

# 全局配置存储
class GlobalConfig:
    def __init__(self):
        self.best_c = None  # 下采样的最优 C
        self.undersample_data = None # 下采样数据
        self.model_config = {}


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