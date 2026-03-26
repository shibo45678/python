# 特征重要性分析（排列重要性、SHAP等）
import copy
from collections import defaultdict

import numpy as np
import pandas as pd


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
    样本 0  [a1, a2, a3] ，样本 1 的是 [a4, a5, a6] 打乱后：
    样本 0 可能得到 [a4, a5, a6]，样本 1 得到 [a1, a2, a3]。
    保留每个样本的时间结构（趋势、周期）但样本之间对应关系被破坏(其他特征和label保持不变）

    优点：保留了该特征的时间自相关性，只破坏了该特征与目标变量之间的关联。
    
    2. SHAP:待实现
    """

    def permutation_importance_lstm(self, model, valsets, output_configs, num_feature_names, cat_feature_names,
                                    model_name, n_repeats=5):
        """计算 LSTM 的排列重要性
            model: 训练好的 LSTM 模型
            valsets: 验证数据 ((num_inputs, cat_input1，cat_input2...), {labels}) / ((num_inputs,), {labels})  TensorSpec
                     解包成 X (3D: [样本, 时间步, 特征]) ； y( [样本，时间步，1] )
            num_feature_names: 传入模型的数值特征
            cat_feature_names: 传入模型的分类特征
            n_repeats: 每个特征重复打乱的次数"""

        if model is None:
            raise ValueError(f'最佳模型没有正确加载')

        feature_names = num_feature_names + cat_feature_names

        X_val, y_val = dataset_to_numpy(dataset=valsets,
                                        cat_columns=cat_feature_names,
                                        output_configs=output_configs)
        feat_num = X_val[0] # ndarray (13961,6,27)
        feat_cat = X_val[1:] # list [ ndarray(13961,6)]

        # 转换后的 NumPy 结构完全可以用于 model.predict，只需以列表形式传入多个输入
        y_pred = model.predict(X_val) # list

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
                    eval = ModelEvaluation(output_configs=output_configs, model_name=model_name)
                    res_old = eval._analyze_binary_classification_task(predictions=y_pred.get(task_name), true_values=true_values,
                                                            task_name=task_name)
                    baseline_metric = res_old.get(metric)
                else:
                    eval = ModelEvaluation(output_configs=output_configs, model_name=model_name)
                    res_old = eval._analyze_multiclass_task(predictions=y_pred.get(task_name),
                                                                       true_values=true_values,
                                                                       task_name=task_name)
                    baseline_metric = res_old.get(metric)

                feat_dict = {}
                j = len(num_feature_names)

                for i, feat_name in enumerate(feature_names):
                    scores = []
                    logger.debug(
                        f"===================================================== {feat_name} 重要性计算 =====================================================")
                    for _ in range(n_repeats):
                        feat_num_copy = copy.deepcopy(feat_num)
                        X_num_permuted = copy.deepcopy(feat_num)  # ndarry

                        # 变动后的“对应特征”的索引数据(0) 是由perm_indices所显示的原索引数据(8)替换
                        perm_indices = np.random.permutation(feat_num_copy.shape[0])

                        if cat_feature_names: # 有分类特征（多+单）
                            feat_cat_copy = copy.deepcopy(feat_cat)
                            X_cat_permuted = copy.deepcopy(feat_cat)

                            if i < j :
                                X_num_permuted[:, :, i] = feat_num_copy[perm_indices, :, i]
                            else:
                                k = i-j
                                X_cat_permuted[k][:, :] = feat_cat_copy[k][perm_indices, :]

                            y_pred_perm = model.predict([X_num_permuted, *X_cat_permuted], verbose=0)
                        else :
                            # 无分类特征
                            X_num_permuted[:, :, i] = feat_num_copy[perm_indices, :, i]
                            y_pred_perm = model.predict([X_num_permuted], verbose=0)

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
                    f'/Users/shibo/AL/NeuralNetwork/temperature_forecasting/data/intermediate/feature_importance_{task_name}')

                # 画图显示
                Visualization.plot_barh(x=importance_sorted['feature'], y=importance_sorted['importance'])
                importance_dict[task_name] = feat_dict
            return importance_dict

def dataset_to_numpy(dataset, cat_columns, output_configs):
    features_list = []
    labels_list = defaultdict(list)
    labels_all = {}

    for batch in dataset:
        if cat_columns:
            features_tuple, labels_dict = batch
            feat_num = [features_tuple[0].numpy()]  # list
            feat_cat = []

            # 处理多分类 # （数值,分类1，分类2）
            for i in range(1, len(cat_columns) + 1):
                feat_cat.append(features_tuple[i].numpy())

            feature_tuple = tuple(feat_num + feat_cat)  # 合并list后，直接转为tuple(list)
            features_list.append(feature_tuple)
        else:
            (feat_num,), labels_dict = batch
            features_list.append((feat_num.numpy(),))
        for task_name in output_configs.keys():
            labels_list[task_name].append(labels_dict.get(task_name))

    # 数值列处理
    feat_num_all = np.concatenate([f[0] for f in features_list], axis=0)  # ndarray (55,6,4)

    # 标签列处理
    for task_name, values_list in labels_list.items():
        labels_all[task_name] = np.concatenate([f for f in values_list], axis=0)

    # 分类列处理
    if cat_columns:
        i = 1
        feat_input = [feat_num_all]
        while i < len(cat_columns) + 1:
            cat = np.concatenate([f[i] for f in features_list], axis=0)  # ndarray (55,6)
            feat_input.append(cat)  # 别赋值
            i += 1
        return feat_input, labels_all # feat_input是list，包括feat_num_all 和 每个cat
    else:
        # 主要解包后别直接array取下角标，这就又下了一个维度，直接feat_num_all不行
        # [feat_num_all] 或者（feat_num_all,) 都可以
        return [feat_num_all], labels_all


# if __name__ == '__main__':
#     df = pd.read_excel('/Users/shibo/AL/NeuralNetwork/temperature_forecasting/data/raw/Workbook1.xlsx')
#     print(df)
#     output_config = {
#         'T': {'type': 'regression',
#               'loss': 'mse',
#               'metrics': ['mae'],
#               'loss_weights': 1,
#               'units': 1,
#               }}
#     window_gen = EnhancedWindowGenerator(
#         mode='train',
#         input_width=6,
#         label_width=5,
#         shift=24,
#         label_columns=['T'],  # 'T','Tpot'
#         numeric_columns=['p', 'T', 'Tpot'],
#         categorical_columns=[],
#         output_configs=output_config,
#         batch_size=16
#     )
#
#     window_data = window_gen.createDataset(df)
#     X_val, y_val = dataset_to_numpy(dataset=window_data, cat_columns=[], output_configs=output_config)
#     feature_names = ['p', 'T', 'Tpot']
#     feat_num = X_val[0]
#     feat_cat = X_val[1:]
#
#
#
#     j =len(['p', 'T', 'Tpot'])
#     for i, feat_name in enumerate(feature_names):
#         feat_num_copy = copy.deepcopy(feat_num)
#         feat_cat_copy = copy.deepcopy(feat_cat)
#
#         X_num_permuted = copy.deepcopy(feat_num) # ndarry
#         X_cat_permuted = copy.deepcopy(feat_cat) # tuple(cat...)
#
#         perm_indices = np.random.permutation(feat_num_copy.shape[0])
#
#         if i < j :
#             X_num_permuted[:, :, i] = feat_num_copy[perm_indices, :, i]
#         else:
#             k = i - j
#             X_cat_permuted[k][:, :] = feat_cat_copy[k][perm_indices, :]



