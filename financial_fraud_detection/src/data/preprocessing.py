import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


def clean_data(data: pd.DataFrame,
               target_column: str = '交易金额') -> pd.DataFrame:
    ''''''
    '''空值'''
    data[target_column].isna().sum()

    '''异常值'''
    q1, q3 = np.quantile(data[target_column], [0.25, 0.75])
    iqr = q3 - q1
    mask_high = data[target_column] >= q3 + iqr * 3
    mask_low = data[target_column] <= q1 - iqr * 3
    data = data.loc[~mask_high]
    data = data.loc[~mask_low]

    '''重复值'''
    data.drop_duplicates(inplace=True)

    return data


'''数据标准化 
"交易金额"，数据波动大，需要标准化 StandardScaler（mean：0 divination：1）；
先划分训练集、测试集，再在训练集里完成数据标准化的fit，后续整体数据的标准化transform，防止数据泄漏'''


def split_data(X: pd.DataFrame,
               y: pd.DataFrame,
               test_size: float = 0.3,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def standardize_data(X_train: pd.DataFrame, X_test: pd.DataFrame,
                     target_column: str = '交易金额',
                     useless_column: str = '时间') -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 无论是 fit 还是 transform 输入要求: array ->values,且二维 -> reshape
    amount_scaler = StandardScaler()

    # 第一步：只在训练集上fit（计算均值和标准差）
    amount_scaler.fit(X_train[target_column].values.reshape(-1, 1))

    # 避免警告
    X_train_standardized = X_train.copy()
    X_test_standardized = X_test.copy()

    # 第二步，分别transform
    X_train_standardized['标准化交易金额'] = amount_scaler.transform(X_train[target_column].values.reshape(-1, 1))
    X_test_standardized['标准化交易金额'] = amount_scaler.transform(X_test[target_column].values.reshape(-1, 1))

    # 将无价值列删除
    X_train = X_train_standardized.drop([target_column, useless_column], axis=1)
    X_test = X_test_standardized.drop([target_column, useless_column], axis=1)

    return X_train, X_test


def undersample(X_train: pd.DataFrame,
                y_train: pd.DataFrame,
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 获取异常值的数量、索引号
    rng = np.random.RandomState(42)  # 使用 RandomState 对象 为choice设置随机数
    number_records_fraud = len(y_train[y_train == 1])
    fraud_indices = np.array(y_train[y_train == 1].index)

    # 保留异常数据，从正常数据里，随机选取和异常数据相同数量的样本（利用索引操作）np.random.choice 转为array
    normal_indices = y_train[y_train == 0].index
    random_normal_indices = rng.choice(normal_indices,  # What to choose from
                                       size=number_records_fraud,  # returns array of * elements
                                       replace=False)  # 相同数据不可重复被选择random sample    默认是True：same item can be chosen multiple times
    random_normal_indices = np.array(random_normal_indices)

    # 将异常样本索引和随机选择的正常样本的索引拼起来，成为下采样样本索引
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    # 根据索引，获取下采样数据集，注意这里一直是原始数据集的索引，需要在原始数据里面检索
    X_undersample = X_train.loc[under_sample_indices]  # 标签索引（index labels） - 更安全
    y_undersample = y_train.loc[under_sample_indices]

    print("下采样样本内，正常样本占比: ", len(y_undersample[y_undersample == 0]) / len(y_undersample))
    print("下采样样本内，异常样本占比: ", len(y_undersample[y_undersample == 1]) / len(y_undersample))
    print("下采样策略总体样本数量: ", len(y_undersample))

    # 对下采样数据集进行切分(调用函数）
    undersample_data = split_data(X_undersample, y_undersample)

    print("下采样 训练集包含样本数量: ", len(undersample_data[0]))
    print("下采样 测试集包含样本数量: ", len(undersample_data[1]))
    print("下采样策略总体样本数量:", len(undersample_data[0]) + len(undersample_data[1]))

    return undersample_data  # 输出split_data的结构 Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]
