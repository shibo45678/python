# 首先，导入所有必要的库
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    PowerTransformer,
    FunctionTransformer
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# 假设您有一个DataFrame df 和目标变量 y
# X = df.drop('target_column', axis=1)
# y = df['target_column']

# 首先，将特征分为[数值型和分类型]
# 请根据您的数据集实际情况修改这些列表
numeric_features = ['age', 'income', 'credit_score']  # 数值型特征列名
categorical_features = ['gender', 'education', 'city']  # 分类型特征列名

# 1. 定义数值型特征的预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 缺失值填充：中位数
    # 选择一种缩放器（取消注释您想要使用的那一个）:
    ('scaler', StandardScaler()),  # 标准化 (均值0, 方差1)
    # ('scaler', MinMaxScaler()),  # 归一化 (到[0,1]范围)
    # ('scaler', RobustScaler()),  # 鲁棒缩放 (对异常值不敏感)
    # ('scaler', PowerTransformer(method='yeo-johnson')),  # 幂变换，使数据更接近正态分布
    # 特征选择（可选）:
    # ('feature_selection', SelectKBest(score_func=f_classif, k=10)),  # 选择前k个最佳特征
    # 降维（可选）:
    # ('pca', PCA(n_components=0.95)),  # 保留95%方差的PCA
])

# 2. 定义分类型特征的预处理管道
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 缺失值填充：常量'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),  # 独热编码，忽略未知类别
])

# 3. 使用ColumnTransformer将不同的预处理应用到不同的特征类型上
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # 处理未被指定的列：'drop'删除, 'passthrough'保留
)

# 4. 选择您想要使用的算法（取消注释一个）
# 注意：某些算法对数据规模敏感（如SVM、KNN、线性模型），需要确保已进行缩放

# 选项A: 随机森林 (通常不需要特别缩放)
# classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 选项B: 梯度提升
# classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 选项C: 逻辑回归 (需要缩放)
classifier = LogisticRegression(random_state=42, max_iter=1000)

# 选项D: 支持向量机 (需要缩放)
# classifier = SVC(random_state=42, probability=True)

# 选项E: K近邻 (需要缩放)
# classifier = KNeighborsClassifier()

# 选项F: 其他算法...
# classifier = AdaBoostClassifier(random_state=42)
# classifier = ExtraTreesClassifier(random_state=42)
# classifier = RidgeClassifier(random_state=42)
# classifier = GaussianNB()

# 5. 构建完整的最终Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # 可以在这里添加全局特征选择（可选）:
    # ('global_feature_selection', RFE(estimator=RandomForestClassifier(n_estimators=50), n_features_to_select=20)),
    ('classifier', classifier)
])

# 6. 使用示例
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练整个管道
full_pipeline.fit(X_train, y_train)

# 进行预测
y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)  # 如果算法支持概率预测

# 评估模型
print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 7. 超参数调优示例（使用GridSearchCV）
# 定义参数网格（注意参数名称格式：步骤名称__参数名称）
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), None],  # 也可以尝试不使用缩放
    'classifier__C': [0.1, 1, 10],  # 逻辑回归的正则化参数
    'classifier__solver': ['liblinear', 'lbfgs']
}

# 创建网格搜索对象
grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 执行网格搜索（这会花费较长时间）
# grid_search.fit(X_train, y_train)

# 查看最佳参数和最佳得分
# print("最佳参数:", grid_search.best_params_)
# print("最佳交叉验证得分:", grid_search.best_score_)

# 使用最佳模型进行预测
# best_model = grid_search.best_estimator_
# y_pred_best = best_model.predict(X_test)

# 8. 保存和加载整个管道（用于部署）
import joblib

# 保存模型
# joblib.dump(full_pipeline, 'my_full_pipeline.pkl')

# 加载模型（在新环境中）
# loaded_pipeline = joblib.load('my_full_pipeline.pkl')
# new_predictions = loaded_pipeline.predict(new_data)