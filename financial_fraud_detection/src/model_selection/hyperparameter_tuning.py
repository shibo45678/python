import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from ..models.logistic_regression import LogisticRegressionModel
from ..evaluation.metrics import evaluate_model_performance
from typing import Union

# 手动循环 正则强度 + KFold
def printing_kfold_scores(x_train_data: pd.DataFrame,
                          y_train_data: Union[pd.Series,pd.DataFrame],
                          n_splits=5) -> float:
    fold = KFold(n_splits=n_splits, shuffle=False)  # 创建 KFold对象（K折交叉验证 K equal parts）

    # 定义不同的正则强度
    c_param_range = [0.01, 0.1, 1, 10, 100]

    # 展示结果表 5行2列 每个强度对应召回系数的均值
    results_table = pd.DataFrame(columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    j = 0  # 外层 遍历不同的正则化强度（**j用于全局表格上索引，i是内部索引，不能混，需要区分）
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('正则化强度: ', c_param)
        print('-------------------------------------------')
        print('')

        # 内层 每个fold的召回率
        lr = LogisticRegressionModel(C=c_param)  # C 外层遍历的正则化强度 其他参数配置相同。L1 正则化、优化算法 liblinear 适合小数据集且L1的情况、random_state(liblinear变）
        recall_accs = []
        for iteration, indices in enumerate(fold.split(x_train_data), start=1):
            # iteration ：i值，第i次交叉验证 ；
            # kfold.split(x_train_data)是个generator,返回2个索引切片 indices (array positions)，train_index 和 test_index
            # 交叉验证训练集的索引 indices[0] ，交叉验证测试集的索引 indices[1] ，利用索引操作比数据本身要好
            # estimator先fit(x，y)学习，y to be a 1D array. 即使1列的df,.values后是2D(n_sample,1),ravel()压成1D；
            # x 不能.values ,转换成array后会丢失features name，但模型需要features name

            # 训练模型：训练集索引indices[0]，获取训练集数据-自变量、因变量
            lr.fit(x_train_data.iloc[indices[0]], y_train_data.iloc[indices[0]].values.ravel())

            # 模型预测：测试集索引 indices[1]，测试集数据-自变量
            y_pred_data = lr.predict(x_train_data.iloc[indices[1]])

            # recall_score(y_actural, y_prediction) 需要“相同数据集”的y_actural 和 y_prediction 做参数。
            # 在y_train_data上，用相同索引获得了“测试集数据的真实y值”
            metrics = evaluate_model_performance(y_train_data.iloc[indices[1]], y_pred_data,plot=False)
            recall_acc = metrics['recall']
            recall_accs.append(recall_acc)  # 5个fold放进list,np.mean()即可求平均
            print('Iteration ', iteration, ': 召回率= ', recall_acc)

        # 执行完所有的交叉验证后，每个参数 c_param 对应的召回率平均值
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('平均召回率: ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].astype(float).idxmax()]['C_parameter']

    print('*********************************************************************************')
    print('效果最好的模型所选参数 = ', best_c)
    print('*********************************************************************************')

    return best_c
