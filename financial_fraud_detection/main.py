from src.data import load_data, plot_class_distribution  # 可参考
from src.data.preprocessing import clean_data, split_data, standardize_data, undersample
from src.models.logistic_regression import LogisticRegressionModel
from src.model_selection.hyperparameter_tuning import printing_kfold_scores
from src.evaluation.metrics import evaluate_model_performance
from src.utils.config import set_config
from src.scripts.undersample_threshold_experiment import UndersampleExperiment


def main():
    data = load_data()

    '''数据均衡性'''
    plot_class_distribution(data['类别'])

    '''清洗'''
    cleaned = clean_data(data)
    X = cleaned.drop(['类别'], axis=1)
    y = cleaned['类别']
    split = split_data(X, y)

    '''切分原始数据集'''
    X_train, X_test = standardize_data(split[0], split[1])
    y_train, y_test = split[2], split[3]
    print("")
    print("原始倾斜数据 训练集包含样本数量: ", len(X_train))
    print("原始倾斜数据 测试集包含样本数量: ", len(X_test))
    print("原始倾斜数据 样本总数: ", len(X_train) + len(X_test))

    '''下采样'''
    # 处理label分类不平衡问题。在训练集里完成下采样,获取下采样后的数据集 undersample_data
    undersample_data = undersample(X_train, y_train)
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = undersample_data

    '''确定模型参数C'''
    best_c = printing_kfold_scores(X_train_undersample, y_train_undersample)

    # 传入全局模型配置，未后续下采样'阈值实验'准备
    set_config(best_c=best_c,
               undersample_data=undersample_data,
               model_config={'penalty': 'l1',
                             'random_state': 42,
                             'solver': 'liblinear',
                             'max_iter': 12000})

    '''训练、测试下采样方案模型'''
    # 用下采样数据的测试集，测试参数确定后的下采样方案模型
    lr = LogisticRegressionModel(C=best_c)
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred_undersample = lr.predict(X_test_undersample)

    '''下采样方案模型评估'''

    print('====下采样方案(下采样测试集)====')
    evaluate_model_performance(y_test_undersample, y_pred_undersample, plot=True)
    # 上述结果解释：这是理想情况下的模型测试结果，因为下采样数据集中，异常样本：正常样本=1:1，原始数据的实际情况是 28W：500。

    print('====下采样方案(原始测试集)====')
    # 用原始数据的测试集，测试“下采样方法”训练出的模型
    y_pred_undersample_model = lr.predict(X_test)
    evaluate_model_performance(y_test, y_pred_undersample_model, plot=True)
    # 结果解释：这是实际情况下的测试，可以看出--下采样数据集上训练处的模型，应用在原始数据量大的数据上，召回率偏差不大。
    # 但实际为正常数据而被预测为异常数据的数据量占比偏高，假阳性变多。FP（误报）。

    print('====欠采样方案直接训练模型====')
    # 如果一开始就用倾斜数据进行模型的训练，而不采用改进的“下采样方法”训练模型，结果会怎样。
    best_c2 = printing_kfold_scores(X_train, y_train)
    lr = LogisticRegressionModel(C=best_c2)
    lr.fit(X_train, y_train.values.ravel())
    y_pred = lr.predict(X_test)
    evaluate_model_performance(y_test, y_pred, plot=True)
    # 结果解释：如果使用原始数据训练模型，测试结果– FP ：假阳性变少，但相对应的召回率较低，很多异常数据没有发现

    print('====调整下采样方案阈值并训练模型====')
    # 原先下采样默认阈值0.5，现在指定阈值 ls.predict_proba()，评估下采样方案各阈值下的模型
    u = UndersampleExperiment()  # 全局 下采样的C和训练预测数据
    u.test_best_thres()
    print('====用新阈值再次预测原始数据====')
    # 看结果调整
    u.run_best_thres(thres=0.7,data_train=X_test,data_true=y_test)








if __name__ == "__main__":
    main()
