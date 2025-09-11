''''''"""
=============================气温预测 - 随机森林回归树=============================
回归树商业价值：
1.对农产品价格预测 
2.制片公司对电影票房预测 
3.网约车平台对用户出行流量预测
"""
import datetime
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor # 集合算法库
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import os

"""1.数据预处理"""
"""1.1 数据集1预处理与可视化"""