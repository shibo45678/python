# 创建并编译一个模型
# original_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, input_shape=(5,)),
#     tf.keras.layers.Dense(1)
# ])
#
# original_model.compile(
#     optimizer='adam',
#     loss='mse',
#     metrics=['mae']
# )
#
# # 保存为 .keras
# original_model.save('test_model.keras')
#
# # 加载模型
# loaded_model = tf.keras.models.load_model('test_model.keras')
#
# # 检查编译状态
# print("原始模型:")
# print(f"  优化器: {original_model.optimizer}")
# print(f"  Loss: {original_model.loss}")
# print(f"  Metrics: {original_model.metrics}")
#
# print("\n加载的模型:")
# print(f"  优化器: {loaded_model.optimizer}")
# print(f"  Loss: {loaded_model.loss}")
# print(f"  Metrics: {loaded_model.metrics}")
#
# # 检查是否可以直接使用
# print(f"\n是否可以直接predict? {'✅' if hasattr(loaded_model, 'predict') else '❌'}")
# print(f"是否可以直接evaluate? {'✅' if hasattr(loaded_model, 'evaluate') else '❌'}")
# print(f"是否可以直接compile? {'✅' if hasattr(loaded_model, 'compile') else '❌'}")
#
# # 验证 metrics 配置
# print(f"\n模型metrics列表: {loaded_model.metrics}")

def get_deployment_model(self):
    """获取部署用的SavedModel路径"""

    if not hasattr(self, 'best_checkpoint'):
        raise ValueError('未找到最佳模型检查点')

    savedmodel_dir = os.path.join(self.best_checkpoint, 'saved_model')

    if not os.path.exists(savedmodel_dir):
        raise FileNotFoundError(
            f"找不到SavedModel目录: {savedmodel_dir}\n"
            "请在训练回调中确保同时保存了SavedModel格式"
        )

    # 验证SavedModel格式
    if not os.path.exists(os.path.join(savedmodel_dir, 'saved_model.pb')):
        raise ValueError(f"不是有效的SavedModel格式: {savedmodel_dir}")

    print(f"✅ 部署模型位置: {savedmodel_dir}")
    return savedmodel_dir


#
# def deploy_with_tensorflow_serving(self):
#     """生成TensorFlow Serving部署命令"""
#
#     savedmodel_dir = self.get_deployment_model()
#
#     # 提取模型名（用于Serving）
#     model_name = os.path.basename(os.path.dirname(savedmodel_dir))
#
#     docker_cmd = f"""
# # TensorFlow Serving 部署命令
# docker run -p 8501:8501 \\
#   --mount type=bind,source={os.path.abspath(savedmodel_dir)},target=/models/{model_name} \\
#   -e MODEL_NAME={model_name} \\
#   -t tensorflow/serving:latest
# """
#
#     logger.debug("=" * 60)
#     logger.debug("TensorFlow Serving 部署命令:")
#     logger.debug("=" * 60)
#     logger.debug(docker_cmd)
#     logger.debug"=" * 60)
#     logger.debug(f"REST API端点: http://localhost:8501/v1/models/{model_name}:predict")
#     logger.debug(f"gRPC端点: localhost:8500")
#     logger.debug("=" * 60)
#
#     return docker_cmd

# def predict_via_savedmodel(self, X):
#     """通过SavedModel预测（测试部署兼容性）"""
#     savedmodel_dir = self.get_deployment_model() # 直接使用保存的SavedModel
#
#     # 加载SavedModel
#     model = tf.saved_model.load(savedmodel_dir)
#     serve_fn = model.signatures['serve']
#
#     # 转换输入格式
#     if isinstance(X, (list, tuple)):
#         # 多输入
#         numeric_input = tf.convert_to_tensor(X[0], dtype=tf.float32)
#         categorical_input = tf.convert_to_tensor(X[1], dtype=tf.float32)
#         result = serve_fn(numeric_input, categorical_input)
#     else:
#         # 单输入
#         result = serve_fn(X)
#
#     return result.numpy()


# def save(self, save_path):
#     """保存整个模型（包括配置、窗口、权重、编译配置）"""
#     check_is_fitted(self)
#     os.makedirs(save_path, exist_ok=True)
#
#     # 1. 保存模型权重 （TF格式，支持大文件）
#     if not hasattr(self, '_prediction_model'):
#         self._prediction_model = self.reconstruct_model()
#
#     # 使用TF格式保存权重（自动分片）
#     weights_dir = os.path.join(save_path, 'model_weights')  # 文件夹放很很多文件
#     self._prediction_model.save_weights(weights_dir)
#
#     # 2. 保存架构为Json
#     model_json = self._prediction_model.to_json()
#     with open(os.path.join(save_path, 'model_architecture.json'), 'w') as f:
#         f.write(model_json)
#
#     # 3. 保存配置信息
#     save_configs = {
#         'model_config': self.model_config,
#         'window_config': {
#             'input_width': self.window.input_width,
#             'label_width': self.window.label_width,
#             'shift': self.window.shift,
#             'label_columns': self.window.label_columns,
#             'numeric_columns': self.window.numeric_columns,
#             'categorical_columns': self.window.categorical_columns,
#             'embedding_configs': self.window.embedding_configs,
#             'output_configs': self.window.output_configs
#         },
#         'compile_config': self._get_compile_config_for_save(),  # 确保字典格式
#         'tensorflow_version': tf.__version__
#     }
#
#     joblib.dump(save_configs, os.path.join(save_path, 'saved_configs.pkl'))
#     logger.debug(f"完整模型已保存到: {save_path}")
#     return save_path

# @classmethod
# def load(cls, save_path):
#     """加载分片保存的模型"""
#
#     # 1. 加载配置
#     config_path = os.path.join(save_path, 'saved_configs.pkl')
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"配置文件不存在: {config_path}")
#
#     saved_configs = joblib.load(config_path)
#
#     # 2. 创建estimator实例
#     estimator = cls(model_config=saved_configs['model_config'])
#
#     # 3. 重建窗口生成器
#     estimator.window = EnhancedWindowGenerator(**saved_configs['window_config'])
#
#     # 4. 从JSON重建模型结构
#     model_json_path = os.path.join(save_path, 'model_architecture.json')
#     if not os.path.exists(model_json_path):
#         raise FileNotFoundError(f"模型架构文件不存在: {model_json_path}")
#
#     with open(model_json_path, 'r') as f:
#         model_json = f.read()
#
#     # 处理自定义层(这里没有)
#     custom_objects = getattr(cls, 'custom_objects', {})
#     estimator.prediction_model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
#
#     # 5. 加载分片权重
#     weights_dir = os.path.join(save_path, 'model_weights')
#     if not os.path.exists(weights_dir):
#         raise FileNotFoundError(f"权重文件不存在: {weights_dir}")
#     # 自动加载所有分片
#     estimator.prediction_model_.load_weights(weights_dir).expect_partial()  # 宽松模式，允许部分权重不匹配
#
#     # 6. 1 重建优化器实例（saved_configs['compile_config']里面保存的是字典，不是实例）'optimizer': {'class_name': 'Adam', 'config': {...}},
#     compile_config = saved_configs['compile_config']
#     optimizer_config = compile_config['optimizer']
#     optimizer_class = getattr(tf.keras.optimizers, optimizer_config['class_name'])
#     optimizer = optimizer_class.from_config(optimizer_config['config'])
#     # 6. 2 提取其他配置
#     loss_config = compile_config['loss']
#     metrics_config = compile_config['metrics']
#     loss_weights_config = compile_config['loss_weights']
#
#     estimator.prediction_model_.compile(
#         optimizer=optimizer,  # 优化器实例
#         loss=loss_config,  # 字典
#         metrics=metrics_config,  # 字典
#         loss_weights=loss_weights_config
#     )
#
#     # 7. 标记为已拟合
#     estimator.is_fitted_ = True
#
#     # training_model_可以为None，因为不需要重新训练
#     estimator.training_model_ = None
#
#     logger.debug(f"模型已从 {save_path} 加载")
#     return estimator


def save_for_deployment(self, deploy_path):
    """ 只保存部署格式 - 从训练检查点复制SavedModel Args: deploy_path: 部署目录 """
    check_is_fitted(self)
    if not hasattr(self, 'best_checkpoint'):
        raise ValueError('需要先训练并保存最佳检查点')

    # 1. 保存模型
    os.makedirs(deploy_path, exist_ok=True)

    source_savedmodel = os.path.join(self.best_checkpoint, 'saved_model')
    target_savedmodel = os.path.join(deploy_path, 'saved_model')

    if not os.path.exists(source_savedmodel):
        logger.warning(f"训练检查点中没有SavedModel，创建新的")
        if not hasattr(self, '_prediction_model'):
            self._prediction_model = self.reconstruct_model()
        self._prediction_model.save(target_savedmodel, save_format='tf')  # export()
    else:
        if os.path.exists(target_savedmodel):
            shutil.rmtree(target_savedmodel)
        shutil.copytree(source_savedmodel, target_savedmodel)
    logger.info(f"SavedModel部署格式：{target_savedmodel}")

    # 2. 保存预处理流水线

    # 保存配置信息
    deploy_config = {
        'model_config': self.model_config,
        'window_config': {
            'input_width': self.window.input_width,
            'label_width': self.window.label_width,
            'shift': self.window.shift,
            'label_columns': self.window.label_columns,
            'numeric_columns': self.window.numeric_columns,
            'categorical_columns': self.window.categorical_columns,
            'embedding_configs': self.window.embedding_configs,
            'output_configs': self.window.output_configs
        }}

    # 保存预测窗口生成器状态
    predict_gen_state = {
        'numeric_indices': self.predict_window_gen.numeric_indices,
        'categorical_indices': self.predict_window_gen.categorical_indices,
        'cat_cols_': self.predict_window_gen.cat_cols_,
        'feature_columns': list(self.predict_window_gen.column_indices.keys()),
        'input_width': self.config['input_width']
    }

    joblib.dump(predict_gen_state, f'{save_path}/predict_gen_state.pkl')

    # 预处理组件
    if hasattr(self, 'scaler'):
        joblib.dump(self.scaler, f'{save_path}/scaler.pkl')

        # # 必需：特征工程配置
        # 'feature_config': {
        #     'input_columns': self._get_input_columns(),
        #     'output_columns': self._get_output_columns(),
        #     'scalers': self._get_scaler_info(),  # 标准化器信息
        #     'encoders': self._get_encoder_info(),  # 编码器信息
        # },
        # 预处理
        # 'data_processing': self._get_data_processing_config(),
        # 'preprocessing': {
        #     'required_columns': self._get_required_columns(),
        #     'normalization': self._get_normalization_info(),
        #

        'deployment_info': {
            'purpose': 'deployment_only',
            'version': '1.0',
            'tensorflow_version': tf.__version__,
            'save_time': datetime.now().isoformat(),
            'training_checkpoint': self.best_checkpoint if hasattr(self, 'best_checkpoint') else None
        }

    }

    with open(os.path.join(deploy_path, 'deploy_config.json'), 'w') as f:
        json.dump(deploy_config, f, indent=2, default=str)

    logger.info(f"✅ 部署包已保存: {deploy_path}")
    logger.info(f"   - SavedModel: {target_savedmodel}")
    logger.info(f"   - 配置: deploy_config.json")

    return deploy_path


@classmethod
def load_for_production(cls, save_path):
    """加载生产环境模型"""
    # 创建estimator实例
    estimator = cls.__new__(cls)

    # 加载模型
    estimator.model = tf.keras.models.load_model(f'{save_path}/model.h5')

    # 加载预测窗口生成器状态
    import joblib
    predict_gen_state = joblib.load(f'{save_path}/predict_gen_state.pkl')

    # 创建预测窗口生成器
    estimator.predict_window_gen = EnhancedWindowGenerator(
        mode='predict',
        input_width=predict_gen_state['input_width'],
        numeric_columns=predict_gen_state['feature_columns'],  # 简化处理
        # 其他参数从state恢复
    )

    # 手动设置索引（因为预测时可能没有原始数据来_setup_column_indices）
    estimator.predict_window_gen.numeric_indices = predict_gen_state['numeric_indices']
    estimator.predict_window_gen.categorical_indices = predict_gen_state['categorical_indices']
    estimator.predict_window_gen.cat_cols_ = predict_gen_state['cat_cols_']
    estimator.predict_window_gen.column_indices = {
        col: i for i, col in enumerate(predict_gen_state['feature_columns'])
    }

    # 加载scaler
    scaler_path = f'{save_path}/scaler.pkl'
    if os.path.exists(scaler_path):
        estimator.scaler = joblib.load(scaler_path)

    return estimator


# def load_for_deployment(deploy_path):
#     """加载部署模型"""
#     savedmodel_dir = os.path.join(deploy_path, 'saved_model')
#     model = tf.keras.models.load_model(savedmodel_dir)
#
#     # 加载部署配置
#     with open(os.path.join(deploy_path, 'deploy_config.json'), 'r') as f:
#         config = json.load(f)
#
#     return model, config


# import numpy as np
# from collections import defaultdict
#
#
# def calculate_mape_with_averaging(predictions, actual_data: pd.DataFrame):
#     """ 单任务处理 包括真实DF
#     正确的方法：对每个时间点的多个预测值取平均，然后计算MAPE
#
#     predictions: 每个窗口的预测结果列表
#                 如: [[105, 108, 112],  # 从t0预测t1,t2,t3
#                      [112, 115, 118],  # 从t1预测t2,t3,t4
#                      ...]
#     actuals: 实际值列表 [100, 110, 105, 120, ...]
#     """
#     # 1. 收集每个时间点的所有预测值
#     predictions_by_time = defaultdict(list)
#
#     for window_start, pred_window in enumerate(predictions):
#         for steps_ahead, pred_value in enumerate(pred_window):
#             target_time = window_start + steps_ahead + 1  # 预测的目标时间点
#
#             time_point = historical_timestamps[target_time]
#             if target_time < len(actuals):
#                 predictions_by_time[time_point].append(pred_value)  # value是表格
#
#     # 2. 时间点维度的mape
#     avg_timepoint_predictions = {}
#     for time_idx, preds in predictions_by_time.items():
#         avg_timepoint_prediction[time_idx] = np.mean(preds)  # list 列表值的多个一起平均
#
#     res1 = calc_level_mape(avg_timepoint_predictions, actuals)
#
#     # 3. 日级别的Mape
#     avg_day_predictions = {}
#     for time_idx, preds in predictions_by_time.items():
#         day = time_idx.dt.day
#         month = time_idx.dt.month
#         year = time_idx.dt.year
#
#         avg_day_predictions[f'{year}_{month}_{day}'] = np.mean(preds)  # list
#
#     # 处理日级别的真实值  actual_data 单任务的带时间的DF
#     actual_data['date'] = actual_data['Date Time'].dt.strptime(format='%Y_%m_%d')
#     daily_actuals = actual_data.groupby('date').agg({'T': 'mean', 'rh': 'mean'})  # task要定
#
#     res2 = calc_level_mape(avg_daily_predictions, daily_actuals)
#
#
# import pandas as pd
#
# ## 示例1：使用字符串时间键
# dates = ['2016-12-31 17:00:00', '2016-12-31 18:00:00', '2016-12-31 19:00:00', '2016-12-31 20:00:00',
#          '2016-12-31 21:00:00', '2016-12-31 22:00:00', '2016-12-31 23:00:00', '2017-01-01 00:00:00']
#
# actuals_dict = {
#     'Date Time': ['2016-12-31 17:00:00', '2016-12-31 18:00:00', '2016-12-31 19:00:00', '2016-12-31 20:00:00',
#                   '2016-12-31 21:00:00', '2016-12-31 22:00:00', '2016-12-31 23:00:00', '2017-01-01 00:00:00'],
#     'T': [1.41, -0.08, -1.03, -1.52, -3.09, -2.59, -3.76, -4.82],
#     'rh': [64.81, 69.81, 70.7, 65.42, 73.7, 71.3, 72.5, 75.7]
# }
# actual = pd.DataFrame(actuals_dict)
# print(actual)
# predictions = [[
#     [3.8703365, 3.884691, 3.4577994, 3.7306015, 2.2956214],
#     [2.6391318, 2.5926297, 2.2178895, 2.5358593, 1.0858217],
#     [1.77491, 1.7106596, 1.1285466, 1.1749766, -0.014756217],
#     [0.88609976, 0.75710475, 0.19265927, 0.11487619, -0.86728024]
# ],
#     [[79.67318, 80.484695, 80.43478, 83.38382, 83.45128],
#      [80.5278, 81.90515, 81.87326, 85.21435, 84.81238, ],
#      [81.605484, 83.30215, 83.313484, 86.7817, 86.10873],
#      [83.98053, 85.9154, 85.84884, 89.42158, 88.54782]]
# ]
# result = calculate_mape_flexible_keys(predictions, actuals_dict)
# print(f"MAPE: {result['mape']:.2f}%")
#
# # 查看详细结果
# for item in result['results_by_time']:
#     print(f"\n时间: {item['time_key']}")
#     print(f"  实际值: {item['actual']}")
#     print(f"  预测次数: {item['n_predictions']}")
#     print(f"  平均预测: {item['avg_prediction']:.2f}")
#     print(f"  APE: {item['ape']:.2f}%")
#
#     # 查看每个预测的来源
#     for detail in item['pred_details']:
#         print(f"    - 从 {detail['window_start']} 预测 {detail['steps_ahead']} 步: {detail['prediction']}")
#
# import numpy as np
#
# # Python 的 and 运算符工作原理：
# result = a and b
# # 等价于：
# if bool(a):
#     result = b
# else:
#     result = a
#
# print("标量运算:")
# # 简单规则：
# # 1. 从左到右检查
# # 2. 遇到第一个为假的，就返回它
# # 3. 如果全部为真，返回最后一个
#
# print(f"3 and 5: {3 and 5}")  # 5（因为 3 为真，返回 5）
# print(f"0 and 5: {0 and 5}")  # 0（因为 0 为假，返回 0）
# print(f"3 and 0: {3 and 0}")  # 0（因为 3 为真，返回 0）
# print(f"False and True: {False and True}")  # False
#
# print({3 and 4 and 5})  # 5
# print({3 and 4 and 6})  # 6
#
# a = pd.Timestamp('2025-02-02')
# print(pd.Timestamp(a.strftime('%Y-%m')))
# rint(f"分钟: {a.floor('T')}")  # 2025-02-02 14:30:00
#
# import pandas as pd
# import numpy as np
#
# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame({
#     'A': ['foo', 'foo', 'bar', 'bar', 'foo'],
#     'B': [1, 2, 3, np.nan, 5],
#     'C': [6, 7, 8, 9, np.nan]
# })
#
# print(df)
# '''
#      A    B    C
# 0  foo  1.0  6.0
# 1  foo  2.0  7.0
# 2  bar  3.0  8.0
# 3  bar  NaN  9.0
# 4  foo  5.0  NaN
# '''
#
# # .size() - 统计每个分组的总行数
# size_result = df.groupby('A').size()
# print(size_result)
# '''
# A
# bar    2  # bar组有2行（索引2,3）
# foo    3  # foo组有3行（索引0,1,4）
# '''
#
# count_result = df.groupby('A').count()
# print(count_result)  # B: 1 3
# count_b = df.groupby('A')['B'].count()  # B  1 3
# print(count_b)
#
# import pandas as pd
# import numpy as np
#
# sales = pd.DataFrame({
#     'Region': ['North', 'North', 'South', 'South', 'North'],
#     'Product': ['A', 'B', 'A', 'A', 'B'],
#     'Sales': [100, 150, 200, np.nan, 120],
#     'Profit': [20, 30, 40, 50, np.nan]
# })
#
# total = sales.groupby('Region').size()  # Series: North 3, South 2
# valid = sales.groupby('Region').count()  # DataFrame
#
# print("total:\n", total)
# print("\nvalid:\n", valid)
# print(total.shape)  # (2,)
# print('\n ', valid.shape)  # (2,3)
#
# # 错误示例 认为可以total（2，）可以直接横向广播
# print("\nvalid / total:\n", valid / total)  # 直接 都是nan 列索引变成： North product profit sales South
#
# # 1. 明确指明行索引对齐
# result = valid.div(total, axis=0)  # 列索引：product profit sales
# print(result)
#
# # 2. 将total.values变成 可以横向广播的 列向量（n,1) ->[:,None] 之后才能正常计算
# print("\nvalid / total.values[:None]", valid / total.values[:, None])
#
# # 3. 或者将total.values 变成可以纵向广播的 行向量(1,n) .reshape / [None,:]
# # 再将valid调整成对应形状（3，2） 即可广播
# print("\nvalid / total.values.reshape(1,-1)", valid.T / total.values.reshape(1, -1))  # (3,2) / (1,2)
#
# # 奇怪的转置 也不知道什么理由？
# print("\nvalid / total", valid.T / total)  # (3,2) / (2,)
#
# import numpy as np
# import pandas as pd
#
# # 创建示例datetime数组
# timepoints = np.array([
#     '2023-01-15T10:30:00',
#     '2023-01-15T14:45:00',
#     '2023-02-20T09:15:00',
#     '2023-02-20T16:20:00',
#     '2024-03-10T11:00:00'
# ], dtype='datetime64[s]')  # 秒精度
#
# # 1. 提取到日（您已经会的）只有array 可以astype,如果仅仅是series，还要用values转换
# dates = timepoints.astype('datetime64[D]')  # YYYY-MM-DD
# # 2. 提取到月
# months = timepoints.astype('datetime64[M]')  # YYYY-MM
# # 3. 提取到年
# years = timepoints.astype('datetime64[Y]')  # YYYY
#
# # 4. 提取到周（ISO周数，更复杂）
# # numpy没有直接的周提取，需要pandas
# weeks = pd.to_datetime(timepoints).isocalendar().year.astype('str') + '-W' + \
#         pd.to_datetime(timepoints).isocalendar().week.astype('str').str.zfill(2)
# # 2023-01-15 10:30:00    2023-W02（+ 文本和文本拼，列表和列表拼）
#
# # 5. 提取到季度
# quarters = pd.to_datetime(timepoints).to_period('Q')  # PeriodIndex(['2023Q1',
# # 6. 提取到小时
# hours = timepoints.astype('datetime64[h]')  # YYYY-MM-DD hh  '2023-01-15T10'
#
# # 7. 更灵活的方法：使用pandas的dt访问器
# timepoints_pd = pd.to_datetime(timepoints)
#
# print("使用pandas提取各种粒度:")
# print(f"年: {timepoints_pd.year.values}")  # 已经是to_datetime()可以直接用.year 不用.dt.year
# print(f"月: {timepoints_pd.month.values}")
# print(f"日: {timepoints_pd.day.values}")
# print(f"小时: {timepoints_pd.hour.values}")
# print(f"分钟: {timepoints_pd.minute.values}")
# print(f"周几(0-6): {timepoints_pd.dayofweek.values}")
# print(f"一年中的第几天: {timepoints_pd.dayofyear.values}")
# print(f"一年中的第几周: {timepoints_pd.isocalendar().week.values}")
#
# arr = np.array(['2023-01-15 10:30:00', '2023-01-15 10:40:00'])
# a = pd.to_datetime(arr)
# # 数组操作
# b = a.values
# c = a[0]
#
# print(b.astype('datetime64[D]'))
# # pandas操作
# print(a.floor('D'))  # 向下取整到日
# print(a.normalize())  # 归一化到日（去掉时分秒）
# print(a.date)  # 提取日期部分（返回datetime.date对象）不用再.dt.date
#
# import pandas as pd
# import numpy as np
#
# # 创建测试数据
# np.random.seed(42)
# n = 50
# timepoints = pd.date_range('2023-01-01', periods=n, freq='h')
# pairs_df = pd.DataFrame({
#     'timepoint': timepoints,
#     'abs_error': np.random.exponential(scale=10, size=n),
#     'squared_error': np.random.exponential(scale=100, size=n)
# })
#
# # 提取日级别
# pairs_df['level'] = pairs_df['timepoint'].dt.floor('D')
#
# # 使用
# daily_stats = pairs_df.groupby('level').agg(
#     mae=('abs_error', 'mean'),
#     mse=('squared_error', lambda x: {
#         'mean': x.mean(),
#         'std': x.std(),
#         'rmse': np.sqrt(x.mean())
#     })
# )
#
# print("结果:")
# print(daily_stats.head())

import cloudpickle
import json
import os
import shutil
from pathlib import Path


def save_for_deployment(self, deploy_path):
    """保存完整部署包"""
    # 检查是否已训练
    check_is_fitted(self)

    # 创建部署目录
    deploy_path = Path(deploy_path)
    deploy_path.mkdir(parents=True, exist_ok=True)

    # 1. 保存模型（从检查点复制 SavedModel）
    source_savedmodel = Path(self.best_checkpoint) / 'saved_model'
    target_savedmodel = deploy_path / 'saved_model'

    if source_savedmodel.exists():
        # 清除目标目录
        if target_savedmodel.exists():
            shutil.rmtree(target_savedmodel)
        # 复制 SavedModel
        shutil.copytree(source_savedmodel, target_savedmodel)
        logger.info(f"已复制 SavedModel: {target_savedmodel}")
    else:
        # 如果没有 SavedModel，创建新的
        if not hasattr(self, '_prediction_model'):
            self._prediction_model = self.reconstruct_model()
        self._prediction_model.save(str(target_savedmodel), save_format='tf')
        logger.info(f"已创建新的 SavedModel: {target_savedmodel}")

    # 2. 保存预处理器（使用 cloudpickle）
    if hasattr(self, 'preprocessor') and self.preprocessor is not None:
        preprocessor_path = deploy_path / 'preprocessor.cpkl'
        with open(preprocessor_path, 'wb') as f:
            cloudpickle.dump(self.preprocessor, f)
        logger.info(f"已保存预处理器: {preprocessor_path}")

    # 3. 保存特征工程管道
    if hasattr(self, 'feature_pipeline') and self.feature_pipeline is not None:
        feature_pipeline_path = deploy_path / 'feature_pipeline.cpkl'
        with open(feature_pipeline_path, 'wb') as f:
            cloudpickle.dump(self.feature_pipeline, f)
        logger.info(f"已保存特征工程管道: {feature_pipeline_path}")

    # 4. 保存后处理器
    if hasattr(self, 'postprocessor') and self.postprocessor is not None:
        postprocessor_path = deploy_path / 'postprocessor.cpkl'
        with open(postprocessor_path, 'wb') as f:
            cloudpickle.dump(self.postprocessor, f)
        logger.info(f"已保存后处理器: {postprocessor_path}")

    # 5. 保存标准化器/编码器
    if hasattr(self, 'scaler') and self.scaler is not None:
        scaler_path = deploy_path / 'scaler.cpkl'
        with open(scaler_path, 'wb') as f:
            cloudpickle.dump(self.scaler, f)
        logger.info(f"已保存标准化器: {scaler_path}")

    # 6. 保存完整的流水线状态（如果之前有保存）
    if hasattr(self, 'serialized_states') and self.serialized_states:
        pipeline_state_path = deploy_path / 'pipeline_states.cpkl'
        with open(pipeline_state_path, 'wb') as f:
            cloudpickle.dump(self.serialized_states, f)
        logger.info(f"已保存完整流水线状态: {pipeline_state_path}")

    # 7. 保存配置和元数据
    config = {
        'model_type': type(self).__name__,
        'input_shape': getattr(self, 'input_shape', None),
        'output_shape': getattr(self, 'output_shape', None),
        'feature_columns': getattr(self, 'feature_columns', None),
        'target_columns': getattr(self, 'target_columns', None),
        'created_at': datetime.now().isoformat(),
        'version': '1.0'
    }

    config_path = deploy_path / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 8. 保存部署包版本信息
    deployment_info = {
        'deployment_format': 'v2',
        'saved_model_path': str(target_savedmodel.relative_to(deploy_path)),
        'components': [],
        'dependencies': self._get_dependencies()
    }

    # 收集所有组件信息
    for file in deploy_path.glob('*.cpkl'):
        deployment_info['components'].append(file.name)

    info_path = deploy_path / 'deployment_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(deployment_info, f, indent=2, ensure_ascii=False)

    logger.info(f"完整部署包已保存到: {deploy_path}")
    return str(deploy_path)


def _get_dependencies(self):
    """获取依赖信息"""
    import tensorflow as tf
    import cloudpickle
    import numpy as np
    import pandas as pd

    return {
        'tensorflow': tf.__version__,
        'cloudpickle': cloudpickle.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__ if 'pd' in locals() else None
    }


class DeploymentModel:
    """部署时使用的模型包装器"""

    def __init__(self, deploy_path):
        self.deploy_path = Path(deploy_path)
        self.loaded = False

    def load(self):
        """加载所有组件"""
        # 1. 加载模型
        saved_model_path = self.deploy_path / 'saved_model'
        if saved_model_path.exists():
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(saved_model_path))
            logger.info(f"已加载 SavedModel: {saved_model_path}")
        else:
            raise FileNotFoundError(f"找不到 SavedModel: {saved_model_path}")

        # 2. 加载预处理器
        preprocessor_path = self.deploy_path / 'preprocessor.cpkl'
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = cloudpickle.load(f)
            logger.info(f"已加载预处理器: {preprocessor_path}")

        # 3. 加载特征工程管道
        feature_pipeline_path = self.deploy_path / 'feature_pipeline.cpkl'
        if feature_pipeline_path.exists():
            with open(feature_pipeline_path, 'rb') as f:
                self.feature_pipeline = cloudpickle.load(f)
            logger.info(f"已加载特征工程管道: {feature_pipeline_path}")

        # 4. 加载后处理器
        postprocessor_path = self.deploy_path / 'postprocessor.cpkl'
        if postprocessor_path.exists():
            with open(postprocessor_path, 'rb') as f:
                self.postprocessor = cloudpickle.load(f)
            logger.info(f"已加载后处理器: {postprocessor_path}")

        # 5. 加载标准化器
        scaler_path = self.deploy_path / 'scaler.cpkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = cloudpickle.load(f)
            logger.info(f"已加载标准化器: {scaler_path}")

        # 6. 加载配置
        config_path = self.deploy_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"已加载配置: {config_path}")

        self.loaded = True
        return self

    def predict(self, input_data):
        """完整的预测流程"""
        if not self.loaded:
            self.load()

        # 1. 预处理
        if hasattr(self, 'preprocessor'):
            processed_data = self.preprocessor.transform(input_data)
        else:
            processed_data = input_data

        # 2. 特征工程
        if hasattr(self, 'feature_pipeline'):
            processed_data = self.feature_pipeline.transform(processed_data)

        # 3. 标准化
        if hasattr(self, 'scaler'):
            processed_data = self.scaler.transform(processed_data)

        # 4. 模型预测
        predictions = self.model.predict(processed_data)

        # 5. 逆标准化
        if hasattr(self, 'scaler'):
            predictions = self.scaler.inverse_transform(predictions)

        # 6. 后处理
        if hasattr(self, 'postprocessor'):
            predictions = self.postprocessor.transform(predictions)

        return predictions

    class ModelDeploymentPackage:
        """创建和管理模型部署包"""

        def __init__(self, trained_model):
            """
            Args:
                trained_model: 已训练好的完整模型对象
            """
            self.trained_model = trained_model

        def create_package(self, output_dir, include_components=None):
            """
            创建部署包

            Args:
                output_dir: 输出目录
                include_components: 要包含的组件列表，如 ['preprocessor', 'scaler', 'postprocessor']
            """
            if include_components is None:
                include_components = ['all']

            package_dir = Path(output_dir) / f"model_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            package_dir.mkdir(parents=True, exist_ok=True)

            # 保存不同组件
            components = {}

            # SavedModel
            if hasattr(self.trained_model, 'model'):
                model_path = package_dir / 'model'
                self.trained_model.model.save(str(model_path), save_format='tf')
                components['model'] = str(model_path)

            # 使用 cloudpickle 保存其他组件
            component_map = {
                'preprocessor': getattr(self.trained_model, 'preprocessor', None),
                'feature_engineer': getattr(self.trained_model, 'feature_engineer', None),
                'scaler': getattr(self.trained_model, 'scaler', None),
                'encoder': getattr(self.trained_model, 'encoder', None),
                'postprocessor': getattr(self.trained_model, 'postprocessor', None),
                'config': getattr(self.trained_model, 'config', {})
            }

            for name, component in component_map.items():
                if component is not None and ('all' in include_components or name in include_components):
                    file_path = package_dir / f"{name}.cpkl"
                    with open(file_path, 'wb') as f:
                        cloudpickle.dump(component, f)
                    components[name] = str(file_path)

            # 保存元数据
            metadata = {
                'created_at': datetime.now().isoformat(),
                'model_type': type(self.trained_model).__name__,
                'components': list(components.keys()),
                'package_version': '1.0'
            }

            metadata_path = package_dir / 'metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"部署包已创建: {package_dir}")
            return package_dir

        #
        # agg(
        #     metrics=('old_col', lambda x: x.mean())
        # )
        #
        # # 创建metrics的多级目录
        # agg(
        #     metrics=('old_col',
        #              lambda x: {
        #                  "mean": np.mean(x),
        #                  "std": np.std(x)}
        #              )
        # )
        # # 作用条件列(转化为字典)
        # cond = {x : 'mean' for x in x.columns if x != 'time_column'}
        # agg(cond)
        #
        # # 字典解包
        # agg(
        #     **{x: self.aggregation for x in x.columns if x !='time_column'}
        # )

        import pandas as pd
        df = pd.DataFrame({
            'abc': ['A', 'A', 'A', 'B', 'B'],
            'num': [1, 2, 3, 10, 20]
        })

        print(df.groupby('abc').agg(percentile=('num', lambda x: pd.cut(x, bins=2).tolist())))

        # 先分桶，再按组和桶分组
        def group_cut(x):
            """在每个分组内独立分桶"""
            return pd.cut(x, bins=2)

        # 应用分组分桶
        df['bucket_correct'] = df.groupby('abc')['num'].transform(group_cut)
        print("\n正确：分组内独立分桶:")
        print(df[['abc', 'num', 'bucket_correct']])
