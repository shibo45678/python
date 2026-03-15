# 特征重要性分析（排列重要性、SHAP等）
from collections import defaultdict

import numpy as np
import pandas as pd
import shap

from data import Visualization
from .model_evaluation import ModelEvaluation
import logging

logger = logging.getLogger(__name__)


class FeatureImportance:
    def __init__(self):
        pass

    """
    1. 排列重要性: (样本数, 时间步数, 特征数)
    评估第 i 个特征，按样本打乱（但保持时间序列完整性）在不同样本之间随机交换。
    样本 A  [a1, a2, a3] ，样本 B 的是 [b1, b2, b3] 打乱后：
    样本 A 可能得到 [b1, b2, b3]，样本 B 得到 [a1, a2, a3]。
    保留每个样本的时间结构（趋势、周期）但样本之间的对应关系被破坏。

    优点：保留了该特征的时间自相关性，只破坏了该特征与目标变量之间的关联（因为特征值现在属于错误的样本）。

    2. SHAP:
    """

    def permutation_importance_lstm(self, model, valsets, output_configs, num_feature_names, cat_feature_names,
                                    model_name, n_repeats=5):
        """计算 LSTM 的排列重要性
            model: 训练好的 LSTM 模型
            valsets: 验证数据 ((num_inputs, cat_input1，cat_input2...), {labels})
                     解包成 X (3D: [样本, 时间步, 特征]) ； y( [样本，时间步，1] )
            num_feature_names: 传入模型的数值特征
            cat_feature_names: 传入模型的分类特征
            n_repeats: 每个特征重复打乱的次数"""

        if model is None:
            raise ValueError(f'最佳模型没有正确加载')

        # feat_num:(样本数, 时间步, 特征数),feat_cat:(样本数, 时间步) 或 (样本数,)
        (feat_num, feat_cat), y_val = dataset_to_numpy(dataset=valsets,
                                                       cat_columns=cat_feature_names,
                                                       output_configs=output_configs)

        # 转换后的 NumPy 结构完全可以用于 model.predict，只需以列表形式传入多个输入
        y_pred = model.predict([feat_num, feat_cat])

        if isinstance(y_val, dict):
            # 获取X_val对应顺序的feature_names() num,cat，label

            importance_dict = {}
            for task_name, true_values in y_val.items():
                task_type = output_configs.get(task_name).get('type')
                metric = output_configs.get(task_name).get('metrics')[0]

                if task_type == 'regression':
                    eval = ModelEvaluation(output_configs=output_configs, model_name=model_name)
                    res_old = eval._analyze_regression_task(predictions=y_pred.get(task_name), true_values=true_values,
                                                            task_name=task_name)
                    baseline_metric = res_old.get(metric)
                elif task_type == 'binary_classification':
                    baseline_metric = 5e-5
                else:
                    baseline_metric = 5e-5

                feat_dict = {}
                for i, feat_name in enumerate(feature_names):
                    scores = []
                    logger.debug(
                        f"                                              {feat_name} 重要性计算                           ")
                    for _ in range(n_repeats):
                        X_num_permuted = feat_num
                        X_cat_permuted = feat_cat

                        perm_indices = np.random.permutation(feat_num.shape[0])

                        if i < len(feature_names) - 1:
                            X_num_permuted[:, :, i] = feat_num[perm_indices, :, i]  # 覆盖
                        else:
                            X_cat_permuted[:, :] = feat_cat[perm_indices, :]

                        y_pred_perm = model.predict([X_num_permuted, X_cat_permuted], verbose=0)

                        eval = ModelEvaluation(output_configs=output_configs, model_name=model_name)
                        res_perm = eval._analyze_regression_task(predictions=y_pred_perm.get(task_name),
                                                                 true_values=true_values,
                                                                 task_name=task_name)
                        perm_metric = res_perm.get(metric)

                        # 重要性 = 误差增加量
                        scores.append(perm_metric - baseline_metric)

                    feat_dict[feat_name] = np.mean(scores)

                # 排序显示
                # feat_df = pd.DataFrame(feat_dict,index=[0])
                # feat_sorted = feat_df.sort_values(0,axis=1,ascending=False)
                # feat_t = feat_sorted.transpose()

                importance_df = pd.DataFrame({
                    'feature': list(feat_dict.keys()),
                    'importance': list(feat_dict.values())
                })
                importance_sorted = importance_df.sort_values('importance', ascending=False)
                importance_sorted.to_csv(
                    f'/Users/shibo/Python/NeuralNetwork/temperature_forecasting/data/intermediate/feature_importance_{task_name}')

                # 画图显示
                Visualization.plot_barh(x=importance_sorted['feature'], y=importance_sorted['importance'])
                importance_dict[task_name] = feat_dict
            return importance_dict


def dataset_to_numpy(dataset,  cat_columns, output_configs):
    features_list = []
    labels_list = defaultdict(list)
    labels_all = {}

    for batch in dataset:
        if cat_columns:
            features_tuple, labels_dict = batch
            feat_num = features_tuple[0]
            feat_cat = []

            # 处理多分类 # （数值,分类1，分类2）
            for i in range(1, len(cat_columns) + 1):
                feat_cat.append(features_tuple[i].numpy())
                feat_cat_tuple = tuple(feat_cat)
                feature_tuple = tuple(feat_num.numpy()) + feat_cat_tuple
                features_list.append(feature_tuple)

        else:
            (feat_num,), labels_dict = batch
            features_list.append((feat_num.numpy(),))

        for task_name in output_configs.keys():
            labels_list[task_name].append(labels_dict.get(task_name))

    # 数值列处理
    feat_num_all = np.concatenate([f[0] for f in features_list], axis=0)

    # 标签列处理
    for task_name, values_list in labels_list.items():
        labels_all[task_name] = np.concatenate([f for f in values_list], axis=0)

    # 分类列处理
    if cat_columns:
        i = 1
        feat_cat_all = tuple()
        while i < len(cat_columns) + 1:
            feat_cat_all = feat_cat_all + (np.concatenate([f[i] for f in features_list], axis=0))
            i += 1
        return (feat_num_all, feat_cat_all), labels_all
    else:
        return (feat_num_all,), labels_all

# if __name__ == '__main__':

