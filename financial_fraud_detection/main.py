from src.data import load_data, plot_class_distribution  # 可参考
from src.data.preprocessing import clean_data, split_data, standardize_data, undersample
from src.models.logistic_regression import LogisticRegressionModel
from src.model_selection.hyperparameter_tuning import printing_kfold_scores
from src.evaluation.metrics import evaluate_model_performance
from src.utils.config import set_config
from src.scripts.undersample_threshold_experiment import UndersampleExperiment
from imblearn.over_sampling import SMOTE  # Imbalanced-Learn 库
import pandas as pd


def main():
    data = load_data()
    """数据均衡性"""
    plot_class_distribution(data['类别'])

    """清洗"""
    cleaned_data = clean_data(data)
    X = cleaned_data.drop(['类别'], axis=1)
    y = cleaned_data['类别']
    split = split_data(X, y)

    """切分原始数据集(标准化前/下采样前）"""
    X_train, X_test = standardize_data(split[0], split[1])
    y_train, y_test = split[2], split[3]
    print("")
    print("原始倾斜数据 训练集包含样本数量: ", len(X_train))
    print("原始倾斜数据 测试集包含样本数量: ", len(X_test))
    print("原始倾斜数据 样本总数: ", len(X_train) + len(X_test))

    """==================================下采样方案=================================="""
    # 处理label分类不平衡问题。在训练集里完成下采样,获取下采样后的数据集 undersample_data
    undersample_data = undersample(X_train, y_train)
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = undersample_data

    """确定模型参数C"""
    best_c = printing_kfold_scores(X_train_undersample, y_train_undersample)

    # 未后续下采样'阈值实验'准备
    set_config(best_c=best_c,  # 0.01
               undersample_data=undersample_data,
               model_config={
                   'penalty': 'l1',
                   'random_state': 42,
                   'solver': 'liblinear',
                   'max_iter': 5000,  # 可以降低，足够让C=0.01收敛
                   'tol': 1e-4  # 可以用更严格的容忍度 标准容忍度
               })

    """训练、测试下采样方案模型"""
    # 用下采样数据的测试集，测试参数确定后的下采样方案模型
    lr = LogisticRegressionModel(best_c)
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred_undersample = lr.predict(X_test_undersample)

    """下采样方案模型评估"""

    print('====下采样方案(下采样测试集)====')
    evaluate_model_performance(y_test_undersample, y_pred_undersample, report=True, plot=True)
    # 上述结果解释：1.这是理想情况下的模型测试召回率结果达到95.65%。因为下采样数据集中，异常样本：正常样本=1:1，原始数据的实际情况是 28W：500。
    # 2.实际为正常数据而被预测为异常数据的数据量占比偏高，假阳性变多。FP（误报）。

    print('====下采样方案(原始测试集)====')
    # 用原始数据的测试集，测试“下采样方法”训练出的模型
    y_pred_undersample_model = lr.predict(X_test)
    evaluate_model_performance(y_test, y_pred_undersample_model, report=True, plot=True)
    # 结果解释：这是实际情况下的测试，可以看出--下采样数据集上训练出的模型，应用在原始数据量大的数据上，召回率偏差不大96%。
    # 同样的FP异常多

    print('====不采样方案直接训练模型====')
    # 如果一开始就用倾斜数据进行模型的训练，而不采用改进的“下采样方法”训练模型，结果会怎样。
    best_c2 = printing_kfold_scores(X_train, y_train)  # 重算最优C
    lr = LogisticRegressionModel(C=best_c2)
    lr.fit(X_train, y_train.values.ravel())
    y_pred = lr.predict(X_test)
    evaluate_model_performance(y_test, y_pred, plot=False)
    # 结果解释：如果使用原始数据训练模型，测试结果– FP ：假阳性变少，但相对应的召回率较低60%，很多异常数据没有发现

    print('====调整下采样方案阈值并训练模型====')
    # 原先下采样默认阈值0.5，现在指定阈值 ls.predict_proba()，评估下采样方案各阈值下的模型
    u = UndersampleExperiment()  # 全局 下采样的C和训练预测数据
    u.test_best_thres()

    # 上述九宫格的结果解释：随着下采样阈值的不断升高，异常判断的标准越来越严格，召回率不断下降。
    # 阈值0.1-0.4时，所有样本几乎都被判断为正例(异常)，没有什么意义，舍弃；
    # 阈值0.5对比0.6, 阈值0.6时，召回率虽有下降，但FP误报人数下降明显，正确检出负例也比较，考虑0.6优于0.5;
    # 阈值0.7时，跟0.6相比，TP正确检出人数相差不大，误报人数略微下降。考虑召回率，优先使用0.6作为阈值，考虑误报人数，优先使用0.7作为阈值；
    # 0.8-0.9 召回率太低，不考虑。

    best_thres = 0.6
    print(f'====用新阈值 {best_thres} 再次预测原始数据====')
    # 看结果调整
    u.run_best_thres(best_thres=best_thres, data_test=X_test, data_true=y_test)

    """==================================SMOTE过采样方案=================================="""

    print('====读取数据、切分数据（同上）====')
    features_train, features_test, labels_train, labels_test = X_train, X_test, y_train, y_test

    print('====SMOTE过采样样本、确定模型最优参数C====')
    # 在训练集中进行过采样样本(SMOTE）自动识别少数类，不需要手动
    oversampler = SMOTE(random_state=42)
    os_features, os_labels = oversampler.fit_resample(features_train, labels_train)
    print(f"原始训练集类别分布:{pd.Series(labels_train).value_counts()}")
    print(f"SMOTE后训练集类别分布:{pd.Series(os_labels).value_counts()}")
    print(f"原始数据形状: {features_train.shape}")
    print(f"过采样后数据形状: {os_features.shape}")

    # 获取最优参数C的值 best_c
    # os_features = pd.DataFrame(os_features)
    # os_labels = pd.DataFrame(os_labels)
    best_c = printing_kfold_scores(os_features, os_labels)

    print('====训练Smote过采样模型，测试原始数据====')
    # 使用平衡后的训练集训练模型,使用原始不平衡测试集去预测模型
    lr = LogisticRegressionModel(C=best_c)
    lr.fit(os_features, os_labels.values.ravel())  # 过采样后数据，去训练模型

    y_pred = lr.predict(features_test)  # 原始数据的测试集预测
    evaluate_model_performance(y_test, y_pred, plot=True, report=True)  # 那这个结果和上面确定阈值的下采样模型预测原始数据的结果比较。
    # 上述结论：使用“过采样SMOTE”训练出的模型 和 使用0.6阈值训练出来的“下采样模型”相比，
    # 在原始测试集数据上的表现，召回率0.912与0.6阈值的0.92相比接近，
    # 但误报FP只有1691，选低于带阈值的测试结果的3907。优先使用Smote过采样模型，最优C=10


if __name__ == "__main__":
    main()
