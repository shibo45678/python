# import matplotlib
#
# matplotlib.use('TkAgg')  # 或 'Qt5Agg', 'MacOSX'（Mac）
#
# import matplotlib.pyplot as plt
#
# # 创建并编译一个模型
# # original_model = tf.keras.Sequential([
# #     tf.keras.layers.Dense(10, input_shape=(5,)),
# #     tf.keras.layers.Dense(1)
# # ])
# #
# # original_model.compile(
# #     optimizer='adam',
# #     loss='mse',
# #     metrics=['mae']
# # )
# #
# # # 保存为 .keras
# # original_model.save('test_model.keras')
# #
# # # 加载模型
# # loaded_model = tf.keras.models.load_model('test_model.keras')
# #
# # # 检查编译状态
# # print("原始模型:")
# # print(f"  优化器: {original_model.optimizer}")
# # print(f"  Loss: {original_model.loss}")
# # print(f"  Metrics: {original_model.metrics}")
# #
# # print("\n加载的模型:")
# # print(f"  优化器: {loaded_model.optimizer}")
# # print(f"  Loss: {loaded_model.loss}")
# # print(f"  Metrics: {loaded_model.metrics}")
# #
# # # 检查是否可以直接使用
# # print(f"\n是否可以直接predict? {'✅' if hasattr(loaded_model, 'predict') else '❌'}")
# # print(f"是否可以直接evaluate? {'✅' if hasattr(loaded_model, 'evaluate') else '❌'}")
# # print(f"是否可以直接compile? {'✅' if hasattr(loaded_model, 'compile') else '❌'}")
# #
# # # 验证 metrics 配置
# # print(f"\n模型metrics列表: {loaded_model.metrics}")
#
# # def deploy_with_tensorflow_serving(self):
# #     """生成TensorFlow Serving部署命令"""
# #
# #     savedmodel_dir = self.get_deployment_model()
# #
# #     # 提取模型名（用于Serving）
# #     model_name = os.path.basename(os.path.dirname(savedmodel_dir))
# #
# #     docker_cmd = f"""
# # # TensorFlow Serving 部署命令
# # docker run -p 8501:8501 \\
# #   --mount type=bind,source={os.path.abspath(savedmodel_dir)},target=/models/{model_name} \\
# #   -e MODEL_NAME={model_name} \\
# #   -t tensorflow/serving:latest
# # """
# #
# #     logger.debug("=" * 60)
# #     logger.debug("TensorFlow Serving 部署命令:")
# #     logger.debug("=" * 60)
# #     logger.debug(docker_cmd)
# #     logger.debug"=" * 60)
# #     logger.debug(f"REST API端点: http://localhost:8501/v1/models/{model_name}:predict")
# #     logger.debug(f"gRPC端点: localhost:8500")
# #     logger.debug("=" * 60)
# #
# #     return docker_cmd
#
# # def predict_via_savedmodel(self, X):
# #     """通过SavedModel预测（测试部署兼容性）"""
# #     savedmodel_dir = self.get_deployment_model() # 直接使用保存的SavedModel
# #
# #     # 加载SavedModel
# #     model = tf.saved_model.load(savedmodel_dir)
# #     serve_fn = model.signatures['serve']
# #
# #     # 转换输入格式
# #     if isinstance(X, (list, tuple)):
# #         # 多输入
# #         numeric_input = tf.convert_to_tensor(X[0], dtype=tf.float32)
# #         categorical_input = tf.convert_to_tensor(X[1], dtype=tf.float32)
# #         result = serve_fn(numeric_input, categorical_input)
# #     else:
# #         # 单输入
# #         result = serve_fn(X)
# #
# #     return result.numpy()
#
#
# # def save(self, save_path):
# #     """保存整个模型（包括配置、窗口、权重、编译配置）"""
# #     check_is_fitted(self)
# #     os.makedirs(save_path, exist_ok=True)
# #
# #     # 1. 保存模型权重 （TF格式，支持大文件）
# #     if not hasattr(self, '_prediction_model'):
# #         self._prediction_model = self.reconstruct_model()
# #
# #     # 使用TF格式保存权重（自动分片）
# #     weights_dir = os.path.join(save_path, 'model_weights')  # 文件夹放很很多文件
# #     self._prediction_model.save_weights(weights_dir)
# #
# #     # 2. 保存架构为Json
# #     model_json = self._prediction_model.to_json()
# #     with open(os.path.join(save_path, 'model_architecture.json'), 'w') as f:
# #         f.write(model_json)
# #
# #     # 3. 保存配置信息
# #     save_configs = {
# #         'model_config': self.model_config,
# #         'window_config': {
# #             'input_width': self.window.input_width,
# #             'label_width': self.window.label_width,
# #             'shift': self.window.shift,
# #             'label_columns': self.window.label_columns,
# #             'numeric_columns': self.window.numeric_columns,
# #             'categorical_columns': self.window.categorical_columns,
# #             'embedding_configs': self.window.embedding_configs,
# #             'output_configs': self.window.output_configs
# #         },
# #         'compile_config': self._get_compile_config_for_save(),  # 确保字典格式
# #         'tensorflow_version': tf.__version__
# #     }
# #
# #     joblib.dump(save_configs, os.path.join(save_path, 'saved_configs.pkl'))
# #     logger.debug(f"完整模型已保存到: {save_path}")
# #     return save_path
#
# # @classmethod
# # def load(cls, save_path):
# #     """加载分片保存的模型"""
# #
# #     # 1. 加载配置
# #     config_path = os.path.join(save_path, 'saved_configs.pkl')
# #     if not os.path.exists(config_path):
# #         raise FileNotFoundError(f"配置文件不存在: {config_path}")
# #
# #     saved_configs = joblib.load(config_path)
# #
# #     # 2. 创建estimator实例
# #     estimator = cls(model_config=saved_configs['model_config'])
# #
# #     # 3. 重建窗口生成器
# #     estimator.window = EnhancedWindowGenerator(**saved_configs['window_config'])
# #
# #     # 4. 从JSON重建模型结构
# #     model_json_path = os.path.join(save_path, 'model_architecture.json')
# #     if not os.path.exists(model_json_path):
# #         raise FileNotFoundError(f"模型架构文件不存在: {model_json_path}")
# #
# #     with open(model_json_path, 'r') as f:
# #         model_json = f.read()
# #
# #     # 处理自定义层(这里没有)
# #     custom_objects = getattr(cls, 'custom_objects', {})
# #     estimator.prediction_model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
# #
# #     # 5. 加载分片权重
# #     weights_dir = os.path.join(save_path, 'model_weights')
# #     if not os.path.exists(weights_dir):
# #         raise FileNotFoundError(f"权重文件不存在: {weights_dir}")
# #     # 自动加载所有分片
# #     estimator.prediction_model_.load_weights(weights_dir).expect_partial()  # 宽松模式，允许部分权重不匹配
# #
# #     # 6. 1 重建优化器实例（saved_configs['compile_config']里面保存的是字典，不是实例）'optimizer': {'class_name': 'Adam', 'config': {...}},
# #     compile_config = saved_configs['compile_config']
# #     optimizer_config = compile_config['optimizer']
# #     optimizer_class = getattr(tf.keras.optimizers, optimizer_config['class_name'])
# #     optimizer = optimizer_class.from_config(optimizer_config['config'])
# #     # 6. 2 提取其他配置
# #     loss_config = compile_config['loss']
# #     metrics_config = compile_config['metrics']
# #     loss_weights_config = compile_config['loss_weights']
# #
# #     estimator.prediction_model_.compile(
# #         optimizer=optimizer,  # 优化器实例
# #         loss=loss_config,  # 字典
# #         metrics=metrics_config,  # 字典
# #         loss_weights=loss_weights_config
# #     )
# #
# #     # 7. 标记为已拟合
# #     estimator.is_fitted_ = True
# #
# #     # training_model_可以为None，因为不需要重新训练
# #     estimator.training_model_ = None
# #
# #     logger.debug(f"模型已从 {save_path} 加载")
# #     return estimator
#
#
# #     # 2. 保存预处理流水线
# #
# #     # 保存配置信息
# #     deploy_config = {
# #         'model_config': self.model_config,
# #         'window_config': {
# #             'input_width': self.window.input_width,
# #             'label_width': self.window.label_width,
# #             'shift': self.window.shift,
# #             'label_columns': self.window.label_columns,
# #             'numeric_columns': self.window.numeric_columns,
# #             'categorical_columns': self.window.categorical_columns,
# #             'embedding_configs': self.window.embedding_configs,
# #             'output_configs': self.window.output_configs
# #         }}
# #
# #     # 保存预测窗口生成器状态
# #     predict_gen_state = {
# #         'numeric_indices': self.predict_window_gen.numeric_indices,
# #         'categorical_indices': self.predict_window_gen.categorical_indices,
# #         'cat_cols_': self.predict_window_gen.cat_cols_,
# #         'feature_columns': list(self.predict_window_gen.column_indices.keys()),
# #         'input_width': self.config['input_width']
# #     }
# #
# #     joblib.dump(predict_gen_state, f'{save_path}/predict_gen_state.pkl')
# #
# #     # 预处理组件
# #     if hasattr(self, 'scaler'):
# #         joblib.dump(self.scaler, f'{save_path}/scaler.pkl')
# #
# #         # # 必需：特征工程配置
# #         # 'feature_config': {
# #         #     'input_columns': self._get_input_columns(),
# #         #     'output_columns': self._get_output_columns(),
# #         #     'scalers': self._get_scaler_info(),  # 标准化器信息
# #         #     'encoders': self._get_encoder_info(),  # 编码器信息
# #         # },
# #         # 预处理
# #         # 'data_processing': self._get_data_processing_config(),
# #         # 'preprocessing': {
# #         #     'required_columns': self._get_required_columns(),
# #         #     'normalization': self._get_normalization_info(),
# #         #
# #
# #         'deployment_info': {
# #             'purpose': 'deployment_only',
# #             'version': '1.0',
# #             'tensorflow_version': tf.__version__,
# #             'save_time': datetime.now().isoformat(),
# #             'training_checkpoint': self.best_checkpoint if hasattr(self, 'best_checkpoint') else None
# #         }
# #
# #     }
# #
# #     with open(os.path.join(deploy_path, 'deploy_config.json'), 'w') as f:
# #         json.dump(deploy_config, f, indent=2, default=str)
# #
# #     logger.info(f"✅ 部署包已保存: {deploy_path}")
# #     logger.info(f"   - SavedModel: {target_savedmodel}")
# #     logger.info(f"   - 配置: deploy_config.json")
# #
# #     return deploy_path
# #
# #
# # @classmethod
# # def load_for_production(cls, save_path):
# #     """加载生产环境模型"""
# #     # 创建estimator实例
# #     estimator = cls.__new__(cls)
# #
# #     # 加载模型
# #     estimator.model = tf.keras.models.load_model(f'{save_path}/model.h5')
# #
# #     # 加载预测窗口生成器状态
# #     import joblib
# #     predict_gen_state = joblib.load(f'{save_path}/predict_gen_state.pkl')
# #
# #     # 创建预测窗口生成器
# #     estimator.predict_window_gen = EnhancedWindowGenerator(
# #         mode='predict',
# #         input_width=predict_gen_state['input_width'],
# #         numeric_columns=predict_gen_state['feature_columns'],  # 简化处理
# #         # 其他参数从state恢复
# #     )
# #
# #     # 手动设置索引（因为预测时可能没有原始数据来_setup_column_indices）
# #     estimator.predict_window_gen.numeric_indices = predict_gen_state['numeric_indices']
# #     estimator.predict_window_gen.categorical_indices = predict_gen_state['categorical_indices']
# #     estimator.predict_window_gen.cat_cols_ = predict_gen_state['cat_cols_']
# #     estimator.predict_window_gen.column_indices = {
# #         col: i for i, col in enumerate(predict_gen_state['feature_columns'])
# #     }
# #
# #     # 加载scaler
# #     scaler_path = f'{save_path}/scaler.pkl'
# #     if os.path.exists(scaler_path):
# #         estimator.scaler = joblib.load(scaler_path)
# #
# #     return estimator
# #
# #
# # # def load_for_deployment(deploy_path):
# # #     """加载部署模型"""
# # #     savedmodel_dir = os.path.join(deploy_path, 'saved_model')
# # #     model = tf.keras.models.load_model(savedmodel_dir)
# # #
# # #     # 加载部署配置
# # #     with open(os.path.join(deploy_path, 'deploy_config.json'), 'r') as f:
# # #         config = json.load(f)
# # #
# # #     return model, config
# #
# #
# # # import numpy as np
# # # from collections import defaultdict
# # #
# # #
# # # def calculate_mape_with_averaging(predictions, actual_data: pd.DataFrame):
# # #     """ 单任务处理 包括真实DF
# # #     正确的方法：对每个时间点的多个预测值取平均，然后计算MAPE
# # #
# # #     predictions: 每个窗口的预测结果列表
# # #                 如: [[105, 108, 112],  # 从t0预测t1,t2,t3
# # #                      [112, 115, 118],  # 从t1预测t2,t3,t4
# # #                      ...]
# # #     actuals: 实际值列表 [100, 110, 105, 120, ...]
# # #     """
# # #     # 1. 收集每个时间点的所有预测值
# # #     predictions_by_time = defaultdict(list)
# # #
# # #     for window_start, pred_window in enumerate(predictions):
# # #         for steps_ahead, pred_value in enumerate(pred_window):
# # #             target_time = window_start + steps_ahead + 1  # 预测的目标时间点
# # #
# # #             time_point = historical_timestamps[target_time]
# # #             if target_time < len(actuals):
# # #                 predictions_by_time[time_point].append(pred_value)  # value是表格
# # #
# # #     # 2. 时间点维度的mape
# # #     avg_timepoint_predictions = {}
# # #     for time_idx, preds in predictions_by_time.items():
# # #         avg_timepoint_prediction[time_idx] = np.mean(preds)  # list 列表值的多个一起平均
# # #
# # #     res1 = calc_level_mape(avg_timepoint_predictions, actuals)
# # #
# # #     # 3. 日级别的Mape
# # #     avg_day_predictions = {}
# # #     for time_idx, preds in predictions_by_time.items():
# # #         day = time_idx.dt.day
# # #         month = time_idx.dt.month
# # #         year = time_idx.dt.year
# # #
# # #         avg_day_predictions[f'{year}_{month}_{day}'] = np.mean(preds)  # list
# # #
# # #     # 处理日级别的真实值  actual_data 单任务的带时间的DF
# # #     actual_data['date'] = actual_data['Date Time'].dt.strptime(format='%Y_%m_%d')
# # #     daily_actuals = actual_data.groupby('date').agg({'T': 'mean', 'rh': 'mean'})  # task要定
# # #
# # #     res2 = calc_level_mape(avg_daily_predictions, daily_actuals)
# # #
# # #
# # # import pandas as pd
# # #
# # # ## 示例1：使用字符串时间键
# # # dates = ['2016-12-31 17:00:00', '2016-12-31 18:00:00', '2016-12-31 19:00:00', '2016-12-31 20:00:00',
# # #          '2016-12-31 21:00:00', '2016-12-31 22:00:00', '2016-12-31 23:00:00', '2017-01-01 00:00:00']
# # #
# # # actuals_dict = {
# # #     'Date Time': ['2016-12-31 17:00:00', '2016-12-31 18:00:00', '2016-12-31 19:00:00', '2016-12-31 20:00:00',
# # #                   '2016-12-31 21:00:00', '2016-12-31 22:00:00', '2016-12-31 23:00:00', '2017-01-01 00:00:00'],
# # #     'T': [1.41, -0.08, -1.03, -1.52, -3.09, -2.59, -3.76, -4.82],
# # #     'rh': [64.81, 69.81, 70.7, 65.42, 73.7, 71.3, 72.5, 75.7]
# # # }
# # # actual = pd.DataFrame(actuals_dict)
# # # print(actual)
# # # predictions = [[
# # #     [3.8703365, 3.884691, 3.4577994, 3.7306015, 2.2956214],
# # #     [2.6391318, 2.5926297, 2.2178895, 2.5358593, 1.0858217],
# # #     [1.77491, 1.7106596, 1.1285466, 1.1749766, -0.014756217],
# # #     [0.88609976, 0.75710475, 0.19265927, 0.11487619, -0.86728024]
# # # ],
# # #     [[79.67318, 80.484695, 80.43478, 83.38382, 83.45128],
# # #      [80.5278, 81.90515, 81.87326, 85.21435, 84.81238, ],
# # #      [81.605484, 83.30215, 83.313484, 86.7817, 86.10873],
# # #      [83.98053, 85.9154, 85.84884, 89.42158, 88.54782]]
# # # ]
# # # result = calculate_mape_flexible_keys(predictions, actuals_dict)
# # # print(f"MAPE: {result['mape']:.2f}%")
# # #
# # # # 查看详细结果
# # # for item in result['results_by_time']:
# # #     print(f"\n时间: {item['time_key']}")
# # #     print(f"  实际值: {item['actual']}")
# # #     print(f"  预测次数: {item['n_predictions']}")
# # #     print(f"  平均预测: {item['avg_prediction']:.2f}")
# # #     print(f"  APE: {item['ape']:.2f}%")
# # #
# # #     # 查看每个预测的来源
# # #     for detail in item['pred_details']:
# # #         print(f"    - 从 {detail['window_start']} 预测 {detail['steps_ahead']} 步: {detail['prediction']}")
# # #
# # # import numpy as np
# # #
# # # # Python 的 and 运算符工作原理：
# # # result = a and b
# # # # 等价于：
# # # if bool(a):
# # #     result = b
# # # else:
# # #     result = a
# # #
# # # print("标量运算:")
# # # # 简单规则：
# # # # 1. 从左到右检查
# # # # 2. 遇到第一个为假的，就返回它
# # # # 3. 如果全部为真，返回最后一个
# # #
# # # print(f"3 and 5: {3 and 5}")  # 5（因为 3 为真，返回 5）
# # # print(f"0 and 5: {0 and 5}")  # 0（因为 0 为假，返回 0）
# # # print(f"3 and 0: {3 and 0}")  # 0（因为 3 为真，返回 0）
# # # print(f"False and True: {False and True}")  # False
# # #
# # # print({3 and 4 and 5})  # 5
# # # print({3 and 4 and 6})  # 6
# # #
# # # a = pd.Timestamp('2025-02-02')
# # # print(pd.Timestamp(a.strftime('%Y-%m')))
# # # rint(f"分钟: {a.floor('T')}")  # 2025-02-02 14:30:00
# # #
# # # import pandas as pd
# # # import numpy as np
# # #
# # # import pandas as pd
# # # import numpy as np
# # #
# # # df = pd.DataFrame({
# # #     'A': ['foo', 'foo', 'bar', 'bar', 'foo'],
# # #     'B': [1, 2, 3, np.nan, 5],
# # #     'C': [6, 7, 8, 9, np.nan]
# # # })
# # #
# # # print(df)
# # # '''
# # #      A    B    C
# # # 0  foo  1.0  6.0
# # # 1  foo  2.0  7.0
# # # 2  bar  3.0  8.0
# # # 3  bar  NaN  9.0
# # # 4  foo  5.0  NaN
# # # '''
# # #
# # # # .size() - 统计每个分组的总行数
# # # size_result = df.groupby('A').size()
# # # print(size_result)
# # # '''
# # # A
# # # bar    2  # bar组有2行（索引2,3）
# # # foo    3  # foo组有3行（索引0,1,4）
# # # '''
# # #
# # # count_result = df.groupby('A').count()
# # # print(count_result)  # B: 1 3
# # # count_b = df.groupby('A')['B'].count()  # B  1 3
# # # print(count_b)
# # #
# # # import pandas as pd
# # # import numpy as np
# # #
# # # sales = pd.DataFrame({
# # #     'Region': ['North', 'North', 'South', 'South', 'North'],
# # #     'Product': ['A', 'B', 'A', 'A', 'B'],
# # #     'Sales': [100, 150, 200, np.nan, 120],
# # #     'Profit': [20, 30, 40, 50, np.nan]
# # # })
# # #
# # # total = sales.groupby('Region').size()  # Series: North 3, South 2
# # # valid = sales.groupby('Region').count()  # DataFrame
# # #
# # # print("total:\n", total)
# # # print("\nvalid:\n", valid)
# # # print(total.shape)  # (2,)
# # # print('\n ', valid.shape)  # (2,3)
# # #
# # # # 错误示例 认为可以total（2，）可以直接横向广播
# # # print("\nvalid / total:\n", valid / total)  # 直接 都是nan 列索引变成： North product profit sales South
# # #
# # # # 1. 明确指明行索引对齐
# # # result = valid.div(total, axis=0)  # 列索引：product profit sales
# # # print(result)
# # #
# # # # 2. 将total.values变成 可以横向广播的 列向量（n,1) ->[:,None] 之后才能正常计算
# # # print("\nvalid / total.values[:None]", valid / total.values[:, None])
# # #
# # # # 3. 或者将total.values 变成可以纵向广播的 行向量(1,n) .reshape / [None,:]
# # # # 再将valid调整成对应形状（3，2） 即可广播
# # # print("\nvalid / total.values.reshape(1,-1)", valid.T / total.values.reshape(1, -1))  # (3,2) / (1,2)
# # #
# # # # 奇怪的转置 也不知道什么理由？
# # # print("\nvalid / total", valid.T / total)  # (3,2) / (2,)
# # #
# # # import numpy as np
# # # import pandas as pd
# # #
# # # # 创建示例datetime数组
# # # timepoints = np.array([
# # #     '2023-01-15T10:30:00',
# # #     '2023-01-15T14:45:00',
# # #     '2023-02-20T09:15:00',
# # #     '2023-02-20T16:20:00',
# # #     '2024-03-10T11:00:00'
# # # ], dtype='datetime64[s]')  # 秒精度
# # #
# # # # 1. 提取到日（您已经会的）只有array 可以astype,如果仅仅是series，还要用values转换
# # # dates = timepoints.astype('datetime64[D]')  # YYYY-MM-DD
# # # # 2. 提取到月
# # # months = timepoints.astype('datetime64[M]')  # YYYY-MM
# # # # 3. 提取到年
# # # years = timepoints.astype('datetime64[Y]')  # YYYY
# # #
# # # # 4. 提取到周（ISO周数，更复杂）
# # # # numpy没有直接的周提取，需要pandas
# # # weeks = pd.to_datetime(timepoints).isocalendar().year.astype('str') + '-W' + \
# # #         pd.to_datetime(timepoints).isocalendar().week.astype('str').str.zfill(2)
# # # # 2023-01-15 10:30:00    2023-W02（+ 文本和文本拼，列表和列表拼）
# # #
# # # # 5. 提取到季度
# # # quarters = pd.to_datetime(timepoints).to_period('Q')  # PeriodIndex(['2023Q1',
# # # # 6. 提取到小时
# # # hours = timepoints.astype('datetime64[h]')  # YYYY-MM-DD hh  '2023-01-15T10'
# # #
# # # # 7. 更灵活的方法：使用pandas的dt访问器
# # # timepoints_pd = pd.to_datetime(timepoints)
# # #
# # # print("使用pandas提取各种粒度:")
# # # print(f"年: {timepoints_pd.year.values}")  # 已经是to_datetime()可以直接用.year 不用.dt.year
# # # print(f"月: {timepoints_pd.month.values}")
# # # print(f"日: {timepoints_pd.day.values}")
# # # print(f"小时: {timepoints_pd.hour.values}")
# # # print(f"分钟: {timepoints_pd.minute.values}")
# # # print(f"周几(0-6): {timepoints_pd.dayofweek.values}")
# # # print(f"一年中的第几天: {timepoints_pd.dayofyear.values}")
# # # print(f"一年中的第几周: {timepoints_pd.isocalendar().week.values}")
# # #
# # # arr = np.array(['2023-01-15 10:30:00', '2023-01-15 10:40:00'])
# # # a = pd.to_datetime(arr)
# # # # 数组操作
# # # b = a.values
# # # c = a[0]
# # #
# # # print(b.astype('datetime64[D]'))
# # # # pandas操作
# # # print(a.floor('D'))  # 向下取整到日
# # # print(a.normalize())  # 归一化到日（去掉时分秒）
# # # print(a.date)  # 提取日期部分（返回datetime.date对象）不用再.dt.date
# # #
# # # import pandas as pd
# # # import numpy as np
# # #
# # # # 创建测试数据
# # # np.random.seed(42)
# # # n = 50
# # # timepoints = pd.date_range('2023-01-01', periods=n, freq='h')
# # # pairs_df = pd.DataFrame({
# # #     'timepoint': timepoints,
# # #     'abs_error': np.random.exponential(scale=10, size=n),
# # #     'squared_error': np.random.exponential(scale=100, size=n)
# # # })
# # #
# # # # 提取日级别
# # # pairs_df['level'] = pairs_df['timepoint'].dt.floor('D')
# # #
# # # # 使用
# # # daily_stats = pairs_df.groupby('level').agg(
# # #     mae=('abs_error', 'mean'),
# # #     mse=('squared_error', lambda x: {
# # #         'mean': x.mean(),
# # #         'std': x.std(),
# # #         'rmse': np.sqrt(x.mean())
# # #     })
# # # )
# # #
# # # print("结果:")
# # # print(daily_stats.head())
# #
# # import cloudpickle
# # import json
# # import os
# # import shutil
# # from pathlib import Path
# #
# #
# # def save_for_deployment(self, deploy_path):
# #     """保存完整部署包"""
# #     # 检查是否已训练
# #     check_is_fitted(self)
# #
# #     # 创建部署目录
# #     deploy_path = Path(deploy_path)
# #     deploy_path.mkdir(parents=True, exist_ok=True)
# #
# #     # 1. 保存模型（从检查点复制 SavedModel）
# #     source_savedmodel = Path(self.best_checkpoint) / 'saved_model'
# #     target_savedmodel = deploy_path / 'saved_model'
# #
# #     if source_savedmodel.exists():
# #         # 清除目标目录
# #         if target_savedmodel.exists():
# #             shutil.rmtree(target_savedmodel)
# #         # 复制 SavedModel
# #         shutil.copytree(source_savedmodel, target_savedmodel)
# #         logger.info(f"已复制 SavedModel: {target_savedmodel}")
# #     else:
# #         # 如果没有 SavedModel，创建新的
# #         if not hasattr(self, '_prediction_model'):
# #             self._prediction_model = self.reconstruct_model()
# #         self._prediction_model.save(str(target_savedmodel), save_format='tf')
# #         logger.info(f"已创建新的 SavedModel: {target_savedmodel}")
# #
# #     # 2. 保存预处理器（使用 cloudpickle）
# #     if hasattr(self, 'preprocessor') and self.preprocessor is not None:
# #         preprocessor_path = deploy_path / 'preprocessor.cpkl'
# #         with open(preprocessor_path, 'wb') as f:
# #             cloudpickle.dump(self.preprocessor, f)
# #         logger.info(f"已保存预处理器: {preprocessor_path}")
# #
# #     # 3. 保存特征工程管道
# #     if hasattr(self, 'feature_pipeline') and self.feature_pipeline is not None:
# #         feature_pipeline_path = deploy_path / 'feature_pipeline.cpkl'
# #         with open(feature_pipeline_path, 'wb') as f:
# #             cloudpickle.dump(self.feature_pipeline, f)
# #         logger.info(f"已保存特征工程管道: {feature_pipeline_path}")
# #
# #     # 4. 保存后处理器
# #     if hasattr(self, 'postprocessor') and self.postprocessor is not None:
# #         postprocessor_path = deploy_path / 'postprocessor.cpkl'
# #         with open(postprocessor_path, 'wb') as f:
# #             cloudpickle.dump(self.postprocessor, f)
# #         logger.info(f"已保存后处理器: {postprocessor_path}")
# #
# #     # 5. 保存标准化器/编码器
# #     if hasattr(self, 'scaler') and self.scaler is not None:
# #         scaler_path = deploy_path / 'scaler.cpkl'
# #         with open(scaler_path, 'wb') as f:
# #             cloudpickle.dump(self.scaler, f)
# #         logger.info(f"已保存标准化器: {scaler_path}")
# #
# #     # 6. 保存完整的流水线状态（如果之前有保存）
# #     if hasattr(self, 'serialized_states') and self.serialized_states:
# #         pipeline_state_path = deploy_path / 'pipeline_states.cpkl'
# #         with open(pipeline_state_path, 'wb') as f:
# #             cloudpickle.dump(self.serialized_states, f)
# #         logger.info(f"已保存完整流水线状态: {pipeline_state_path}")
# #
# #     # 7. 保存配置和元数据
# #     config = {
# #         'model_type': type(self).__name__,
# #         'input_shape': getattr(self, 'input_shape', None),
# #         'output_shape': getattr(self, 'output_shape', None),
# #         'feature_columns': getattr(self, 'feature_columns', None),
# #         'target_columns': getattr(self, 'target_columns', None),
# #         'created_at': datetime.now().isoformat(),
# #         'version': '1.0'
# #     }
# #
# #     config_path = deploy_path / 'config.json'
# #     with open(config_path, 'w', encoding='utf-8') as f:
# #         json.dump(config, f, indent=2, ensure_ascii=False)
# #
# #     # 8. 保存部署包版本信息
# #     deployment_info = {
# #         'deployment_format': 'v2',
# #         'saved_model_path': str(target_savedmodel.relative_to(deploy_path)),
# #         'components': [],
# #         'dependencies': self._get_dependencies()
# #     }
# #
# #     # 收集所有组件信息
# #     for file in deploy_path.glob('*.cpkl'):
# #         deployment_info['components'].append(file.name)
# #
# #     info_path = deploy_path / 'deployment_info.json'
# #     with open(info_path, 'w', encoding='utf-8') as f:
# #         json.dump(deployment_info, f, indent=2, ensure_ascii=False)
# #
# #     logger.info(f"完整部署包已保存到: {deploy_path}")
# #     return str(deploy_path)
# #
# #
# # def _get_dependencies(self):
# #     """获取依赖信息"""
# #     import tensorflow as tf
# #     import cloudpickle
# #     import numpy as np
# #     import pandas as pd
# #
# #     return {
# #         'tensorflow': tf.__version__,
# #         'cloudpickle': cloudpickle.__version__,
# #         'numpy': np.__version__,
# #         'pandas': pd.__version__ if 'pd' in locals() else None
# #     }
# #
# #
# # class DeploymentModel:
# #     """部署时使用的模型包装器"""
# #
# #     def __init__(self, deploy_path):
# #         self.deploy_path = Path(deploy_path)
# #         self.loaded = False
# #
# #     def load(self):
# #         """加载所有组件"""
# #         # 1. 加载模型
# #         saved_model_path = self.deploy_path / 'saved_model'
# #         if saved_model_path.exists():
# #             import tensorflow as tf
# #             self.model = tf.keras.models.load_model(str(saved_model_path))
# #             logger.info(f"已加载 SavedModel: {saved_model_path}")
# #         else:
# #             raise FileNotFoundError(f"找不到 SavedModel: {saved_model_path}")
# #
# #         # 2. 加载预处理器
# #         preprocessor_path = self.deploy_path / 'preprocessor.cpkl'
# #         if preprocessor_path.exists():
# #             with open(preprocessor_path, 'rb') as f:
# #                 self.preprocessor = cloudpickle.load(f)
# #             logger.info(f"已加载预处理器: {preprocessor_path}")
# #
# #         # 3. 加载特征工程管道
# #         feature_pipeline_path = self.deploy_path / 'feature_pipeline.cpkl'
# #         if feature_pipeline_path.exists():
# #             with open(feature_pipeline_path, 'rb') as f:
# #                 self.feature_pipeline = cloudpickle.load(f)
# #             logger.info(f"已加载特征工程管道: {feature_pipeline_path}")
# #
# #         # 4. 加载后处理器
# #         postprocessor_path = self.deploy_path / 'postprocessor.cpkl'
# #         if postprocessor_path.exists():
# #             with open(postprocessor_path, 'rb') as f:
# #                 self.postprocessor = cloudpickle.load(f)
# #             logger.info(f"已加载后处理器: {postprocessor_path}")
# #
# #         # 5. 加载标准化器
# #         scaler_path = self.deploy_path / 'scaler.cpkl'
# #         if scaler_path.exists():
# #             with open(scaler_path, 'rb') as f:
# #                 self.scaler = cloudpickle.load(f)
# #             logger.info(f"已加载标准化器: {scaler_path}")
# #
# #         # 6. 加载配置
# #         config_path = self.deploy_path / 'config.json'
# #         if config_path.exists():
# #             with open(config_path, 'r', encoding='utf-8') as f:
# #                 self.config = json.load(f)
# #             logger.info(f"已加载配置: {config_path}")
# #
# #         self.loaded = True
# #         return self
# #
# #     def predict(self, input_data):
# #         """完整的预测流程"""
# #         if not self.loaded:
# #             self.load()
# #
# #         # 1. 预处理
# #         if hasattr(self, 'preprocessor'):
# #             processed_data = self.preprocessor.transform(input_data)
# #         else:
# #             processed_data = input_data
# #
# #         # 2. 特征工程
# #         if hasattr(self, 'feature_pipeline'):
# #             processed_data = self.feature_pipeline.transform(processed_data)
# #
# #         # 3. 标准化
# #         if hasattr(self, 'scaler'):
# #             processed_data = self.scaler.transform(processed_data)
# #
# #         # 4. 模型预测
# #         predictions = self.model.predict(processed_data)
# #
# #         # 5. 逆标准化
# #         if hasattr(self, 'scaler'):
# #             predictions = self.scaler.inverse_transform(predictions)
# #
# #         # 6. 后处理
# #         if hasattr(self, 'postprocessor'):
# #             predictions = self.postprocessor.transform(predictions)
# #
# #         return predictions
# #
# #     class ModelDeploymentPackage:
# #         """创建和管理模型部署包"""
# #
# #         def __init__(self, trained_model):
# #             """
# #             Args:
# #                 trained_model: 已训练好的完整模型对象
# #             """
# #             self.trained_model = trained_model
# #
# #         def create_package(self, output_dir, include_components=None):
# #             """
# #             创建部署包
# #
# #             Args:
# #                 output_dir: 输出目录
# #                 include_components: 要包含的组件列表，如 ['preprocessor', 'scaler', 'postprocessor']
# #             """
# #             if include_components is None:
# #                 include_components = ['all']
# #
# #             package_dir = Path(output_dir) / f"model_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# #             package_dir.mkdir(parents=True, exist_ok=True)
# #
# #             # 保存不同组件
# #             components = {}
# #
# #             # SavedModel
# #             if hasattr(self.trained_model, 'model'):
# #                 model_path = package_dir / 'model'
# #                 self.trained_model.model.save(str(model_path), save_format='tf')
# #                 components['model'] = str(model_path)
# #
# #             # 使用 cloudpickle 保存其他组件
# #             component_map = {
# #                 'preprocessor': getattr(self.trained_model, 'preprocessor', None),
# #                 'feature_engineer': getattr(self.trained_model, 'feature_engineer', None),
# #                 'scaler': getattr(self.trained_model, 'scaler', None),
# #                 'encoder': getattr(self.trained_model, 'encoder', None),
# #                 'postprocessor': getattr(self.trained_model, 'postprocessor', None),
# #                 'config': getattr(self.trained_model, 'config', {})
# #             }
# #
# #             for name, component in component_map.items():
# #                 if component is not None and ('all' in include_components or name in include_components):
# #                     file_path = package_dir / f"{name}.cpkl"
# #                     with open(file_path, 'wb') as f:
# #                         cloudpickle.dump(component, f)
# #                     components[name] = str(file_path)
# #
# #             # 保存元数据
# #             metadata = {
# #                 'created_at': datetime.now().isoformat(),
# #                 'model_type': type(self.trained_model).__name__,
# #                 'components': list(components.keys()),
# #                 'package_version': '1.0'
# #             }
# #
# #             metadata_path = package_dir / 'metadata.json'
# #             with open(metadata_path, 'w', encoding='utf-8') as f:
# #                 json.dump(metadata, f, indent=2)
# #
# #             logger.info(f"部署包已创建: {package_dir}")
# #             return package_dir
# #
# #         #
# #         # agg(
# #         #     metrics=('old_col', lambda x: x.mean())
# #         # )
# #         #
# #         # # 创建metrics的多级目录
# #         # agg(
# #         #     metrics=('old_col',
# #         #              lambda x: {
# #         #                  "mean": np.mean(x),
# #         #                  "std": np.std(x)}
# #         #              )
# #         # )
# #         # # 作用条件列(转化为字典)
# #         # cond = {x : 'mean' for x in x.columns if x != 'time_column'}
# #         # agg(cond)
# #         #
# #         # # 字典解包
# #         # agg(
# #         #     **{x: self.aggregation for x in x.columns if x !='time_column'}
# #         # )
# #
# #         import pandas as pd
# #         df = pd.DataFrame({
# #             'abc': ['A', 'A', 'A', 'B', 'B'],
# #             'num': [1, 2, 3, 10, 20]
# #         })
# #
# #         print(df.groupby('abc').agg(percentile=('num', lambda x: pd.cut(x, bins=2).tolist())))
# #
# #         # 先分桶，再按组和桶分组
# #         def group_cut(x):
# #             """在每个分组内独立分桶"""
# #             return pd.cut(x, bins=2)
# #
# #         # 应用分组分桶
# #         df['bucket_correct'] = df.groupby('abc')['num'].transform(group_cut)
# #         print("\n正确：分组内独立分桶:")
# #         print(df[['abc', 'num', 'bucket_correct']])
# #
# #
# #
# #
# #         if not hasattr(self, 'encoders_') or self.encoders_ is None:
# #             logger.warning("没有找到 encoders_ 信息，请先fit")
# #             return scaled_data
# #
# #         # 概率数组 argmax 得到类别索引，变成二维数组
# #         label_indices = np.argmax(scaled_data, axis=-1)  # (batch, 5)
# #
# #         unknown_encoded = None
# #         if hasattr(self, 'unknown_token_map_') and target_column in self.unknown_token_map_:
# #             unknown_encoded = self.unknown_token_map_[target_column]
# #
# #         if unknown_encoded is not None:
# #             unknown_mask = (label_indices == unknown_encoded)
# #
# #             # 临时替换unknown 为有效（避免编码器报错）
# #             label_indices_clean = label_indices.copy()
# #             label_indices_clean[unknown_mask] = 0
# #         else:
# #             label_indices_clean = label_indices
# #
# #         # 整体处理，再转回
# #         flat_indices = label_indices_clean.flatten()
# #         encoder = self.encoders_[target_column]
# #
# #         decoded = encoder.inverse_transfrom(flat_indices)
# #         result = decoded.reshape(label_indices.shape)
# #
# #         if unknown_encoded is not None:
# #             unknown_mask = (label_indices == unknown_encoded)
# #             result = result.astype(object)  # 允许None
# #             result[unknown_mask] = None
# #
# #         return result
# """偏度"""
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.stats import norm, skewnorm, kurtosis
# #
# # # 创建数据
# # data_normal = np.random.normal(0, 1, 1000)
# # data_skew_pos = skewnorm.rvs(5, size=1000)  # 正偏
# # data_skew_neg = skewnorm.rvs(-5, size=1000)  # 负偏
# #
# # # 绘制直方图
# # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# #
# # axs[0].hist(data_normal, bins=50, alpha=0.7, color='blue', density=True)
# # axs[0].set_title('Normal Distribution')
# #
# # axs[1].hist(data_skew_pos, bins=50, alpha=0.7, color='orange', density=True)
# # axs[1].set_title('Positive Skew')
# #
# # axs[2].hist(data_skew_neg, bins=50, alpha=0.7, color='green', density=True)
# # axs[2].set_title('Negative Skew')
# #
# # plt.show()
#
# """峰度"""
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from scipy import stats
#
# matplotlib.set_loglevel('warning')
# plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy import stats
# #
# # # 设置随机种子以保证结果可复现
# # np.random.seed(42)
# #
# # # 创建子图
# # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# #
# # # 1. 均匀分布：轻尾分布 (超额峰度 < 0)
# # uniform_data = np.random.uniform(-3, 3, 10000)
# # kurt_uniform = stats.kurtosis(uniform_data)
# # axes[0].hist(uniform_data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
# # x_uniform = np.linspace(-3, 3, 100)
# # y_uniform = stats.uniform.pdf(x_uniform, loc=-3, scale=6)  # uniform(-3, 3) 的 PDF 是 1/6
# # axes[0].plot(x_uniform, y_uniform, 'r-', linewidth=2, label='PDF')
# # axes[0].set_title(f'轻尾分布\n超额峰度 = {kurt_uniform:.2f}')
# # axes[0].set_xlabel('Value')
# # axes[0].set_ylabel('Density')
# # axes[0].legend()
# #
# # # 2. 正态分布：峰度 ≈ 0
# # normal_data = np.random.normal(0, 1, 10000)
# # kurt_normal = stats.kurtosis(normal_data)
# # axes[1].hist(normal_data, bins=50, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
# # x_norm = np.linspace(-4, 4, 100)
# # y_norm = stats.norm.pdf(x_norm, 0, 1)
# # axes[1].plot(x_norm, y_norm, 'r-', linewidth=2, label='PDF')
# # axes[1].set_title(f'正态分布\n超额峰度 = {kurt_normal:.2f}')
# # axes[1].set_xlabel('Value')
# # axes[1].set_ylabel('Density')
# # axes[1].legend()
# #
# # # 3. t 分布（df=3）：重尾分布 (超额峰度 > 0)
# # t_data = np.random.standard_t(3, 10000) * 0.7  # 缩放使均值为0，方差接近1
# # kurt_t = stats.kurtosis(t_data)
# # axes[2].hist(t_data, bins=100, range=(-5, 5), alpha=0.7, density=True, color='salmon', edgecolor='black')
# # axes[2].plot(x_norm, y_norm, 'r-', linewidth=2, alpha=0.5, label='Normal PDF')
# # axes[2].set_title(f'重尾分布\n超额峰度 = {kurt_t:.2f}')
# # axes[2].set_xlabel('Value')
# # axes[2].set_ylabel('Density')
# # axes[2].legend()
# #
# # # 统一坐标轴范围以便比较
# # for ax in axes:
# #     ax.set_xlim(-5, 5)
# #     ax.grid(True, alpha=0.3)
# #
# # plt.tight_layout()
# # plt.show()
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.stats import norm, t
# #
# # x = np.linspace(-5, 5, 1000)
# # plt.plot(x, norm.pdf(x), label='N(0,1)', lw=2)
# #
# # for df in [1, 2, 5, 10, 30]:
# #     plt.plot(x, t.pdf(x, df), label=f't (df={df})', linestyle='--')
# #
# # plt.ylim(0, 0.5)
# # plt.legend()
# # plt.title('t 分布 vs 标准正态分布')
# # plt.show()
#
# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyBboxPatch
#
# # 支持中文显示（Windows / macOS / Linux 通用）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
#
# # fig, ax = plt.subplots(figsize=(8, 10))
# # ax.set_xlim(0, 10)
# # ax.set_ylim(0, 12)
# # ax.axis('off')
# #
# # # 节点样式
# # def add_box(x, y, text, width=6, height=0.8, color='lightblue'):
# #     box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3", edgecolor='black', facecolor=color)
# #     ax.add_patch(box)
# #     ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=12)
# #
# # # 连线函数
# # def add_arrow(x1, y1, x2, y2, label="", offset=0):
# #     ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
# #                 arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
# #     if label:
# #         ax.text((x1+x2)/2 + offset, (y1+y2)/2 + 0.1, label, ha='center', fontsize=11)
# #
# # # 绘制节点（从上到下）
# # add_box(2, 10.5, "开始")
# # add_box(1.5, 9, "你知道总体标准差 σ 吗？")
# # add_box(0, 7, "是（罕见）", color='lightgreen')
# # add_box(5, 7, "否（常见！）", color='lightcoral')
# # add_box(0, 5, "样本量 n ≥ 30？", color='lightyellow')
# # add_box(5, 5, "样本量 n ≥ 30？", color='lightyellow')
# # add_box(-0.5, 3, "用 Z 检验\n（σ 已知）", color='lightgreen')
# # add_box(2, 3, "用 Z 检验\n（不推荐！n<30 且 σ 已知极少见）", color='orange')
# # add_box(4.5, 3, "用 t 检验\n（t ≈ Z，但更规范）", color='lightblue')
# # add_box(7, 3, "✅ 用 t 检验\n（小样本 + σ 未知）", color='lightblue')
# #
# # # 连线
# # add_arrow(5, 10.5, 4.5, 9.4)  # 开始 → 问题
# # add_arrow(4.5, 8.9, 2.5, 7.4, "是")
# # add_arrow(4.5, 8.9, 5.5, 7.4, "否")
# #
# # # “是”分支
# # add_arrow(2.5, 6.9, 2.5, 5.4)
# # add_arrow(2.5, 4.9, 1.5, 3.4, "是")
# # add_arrow(2.5, 4.9, 3.5, 3.4, "否")
# #
# # # “否”分支
# # add_arrow(5.5, 6.9, 5.5, 5.4)
# # add_arrow(5.5, 4.9, 6.0, 3.4, "是")
# # add_arrow(5.5, 4.9, 8.0, 3.4, "否")
# #
# # # 标题
# # ax.text(5, 11.3, "t 检验 vs Z 检验 决策流程图", ha='center', fontsize=16, weight='bold')
# #
# # # 保存图片
# # plt.tight_layout()
# # plt.savefig("t_test_decision_flowchart.png", dpi=300, bbox_inches='tight')
# # plt.show()
# #
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.stats import norm, t
# #
# # # 设置中文字体支持（适用于 Windows / macOS / Linux）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
#
# # # 定义 x 轴范围（覆盖 ±5）
# # x = np.linspace(-5, 5, 1000)
# #
# # # 计算各分布的概率密度
# # y_normal = norm.pdf(x)  # 正态分布
# #
# # # t 分布，不同自由度
# # df_list = [1, 2, 5, 30]
# # y_t = [t.pdf(x, df) for df in df_list]
# #
# # # 创建图形
# # fig, ax = plt.subplots(figsize=(10, 6))
# #
# # # 绘制正态分布
# # ax.plot(x, y_normal, label='标准正态分布 (N(0,1))', color='black', linewidth=2)
# #
# # # 绘制 t 分布
# # colors = ['red', 'orange', 'blue', 'green']
# # labels = [f't 分布 (ν={df})' for df in df_list]
# #
# # for i, (y, color, label) in enumerate(zip(y_t, colors, labels)):
# #     ax.plot(x, y, label=label, color=color, linewidth=2)
# #
# # # 图形美化
# # ax.set_title('t 分布 vs 标准正态分布：不同自由度对比', fontsize=16, pad=20)
# # ax.set_xlabel('x', fontsize=12)
# # ax.set_ylabel('概率密度', fontsize=12)
# # ax.grid(True, alpha=0.3)
# # ax.legend(loc='upper right', fontsize=10)
# #
# # # 设置坐标轴范围
# # ax.set_xlim(-5, 5)
# # ax.set_ylim(0, 0.5)
# #
# # # 保存图片（高清）
# # plt.tight_layout()
# # plt.savefig("t_distribution_comparison.png", dpi=300, bbox_inches='tight')
# # plt.show()
# #
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.stats import norm, t
# #
# # # 创建x轴范围
# # x = np.linspace(-4, 4, 1000)
# #
# # # 正态分布
# # normal_pdf = norm.pdf(x)
# #
# # # t分布（自由度=3，具有厚尾特性）
# # t_pdf = t.pdf(x, df=3)
# #
# # # 计算t分布的峰度（理论值）
# # # t分布的理论峰度 = 6/(df-4) for df>4
# # df = 3
# # # 当df<=4时，峰度未定义（无穷大），这与您的32.56相符
# #
# # plt.figure(figsize=(10, 6))
# #
# # # 绘制两条曲线
# # plt.plot(x, normal_pdf, 'b-', linewidth=2, label='Normal PDF (峰度=0)')
# # plt.plot(x, t_pdf, 'r--', linewidth=2, label=f't分布 (df={df}, 厚尾)')
# #
# # plt.xlabel('Value', fontsize=12)
# # plt.ylabel('Density', fontsize=12)
# # plt.title('正态分布 vs t分布：厚尾对比', fontsize=14)
# # plt.legend(fontsize=12)
# # plt.grid(True, alpha=0.3)
# #
# # # 标记尾部区域
# # plt.fill_between(x[x < -2], 0, normal_pdf[x < -2], alpha=0.2, color='blue')
# # plt.fill_between(x[x < -2], 0, t_pdf[x < -2], alpha=0.2, color='red')
# # plt.fill_between(x[x > 2], 0, normal_pdf[x > 2], alpha=0.2, color='blue')
# # plt.fill_between(x[x > 2], 0, t_pdf[x > 2], alpha=0.2, color='red')
# #
# # plt.text(-3, 0.05, 't分布尾部更厚\n(异常值概率更高)',
# #          fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
# #
# # plt.show()
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.stats import norm, t
# #
# # # 创建x轴范围
# # x = np.linspace(-5, 5, 1000)
# #
# # # 标准正态分布
# # normal_pdf = norm.pdf(x)
# #
# # # t分布（自由度=5）
# # df = 5
# # # t分布的原始方差 = df/(df-2) = 5/3 ≈ 1.667
# # # 缩放因子使方差=1：缩放因子 = sqrt((df-2)/df) = sqrt(3/5)
# # scale_factor = np.sqrt((df-2)/df)
# # t_scaled_pdf = t.pdf(x/scale_factor, df) / scale_factor
# #
# # plt.figure(figsize=(12, 6))
# # plt.plot(x, normal_pdf, 'b-', linewidth=2, label='标准正态分布 (方差=1, 峰度=0)')
# # plt.plot(x, t_scaled_pdf, 'r--', linewidth=2, label=f't分布(df={df}, 方差=1, 峰度={6/(df-4):.2f})')
# #
# # plt.xlabel('Value')
# # plt.ylabel('密度函数')
# # plt.title('相同方差下的分布比较：t分布既"尖峰"又"厚尾"')
# # plt.legend()
# # plt.grid(True, alpha=0.3)
# #
# # # 标记中心区域
# # plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
# # plt.text(0.1, 0.42, 't分布中心更陡峭\n峰值更高', fontsize=10)
# #
# # # 标记尾部区域
# # plt.fill_between(x[x < -2], 0, normal_pdf[x < -2], alpha=0.2, color='blue')
# # plt.fill_between(x[x < -2], 0, t_scaled_pdf[x < -2], alpha=0.2, color='red')
# # plt.text(-3, 0.02, 't分布尾部更厚重\n异常值概率更高', fontsize=10)
# #
# # plt.show()
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.stats import norm, t
# #
# # # 创建对比
# # x = np.linspace(-5, 5, 1000)
# #
# # # 情况1：不同的正态分布（方差不同）
# # normal_var1 = norm.pdf(x, 0, 1)      # 方差=1
# # normal_var2 = norm.pdf(x, 0, 2)      # 方差=4（尺度参数=2）
# #
# # # 情况2：正态分布 vs t分布
# # normal = norm.pdf(x, 0, 1)
# # t_dist = t.pdf(x, df=3)  # 方差=3/(3-2)=3
# #
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# #
# # # 左侧：不同方差的正态分布比较
# # ax1.plot(x, normal_var1, 'b-', label='N(0,1) 方差=1')
# # ax1.plot(x, normal_var2, 'g-', label='N(0,2²) 方差=4')
# # ax1.set_title('正态分布族：方差越大越平坦')
# # ax1.legend()
# # ax1.grid(alpha=0.3)
# #
# # # 右侧：相同方差尺度的比较
# # ax2.plot(x, normal, 'b-', label=f'N(0,1) 方差=1')
# # # 为了公平比较，我们缩放t分布使其方差=1
# # scale_factor = np.sqrt(1/3)  # 因为t(3)方差=3，要使其方差=1，需要除以√3
# # t_scaled = t.pdf(x/scale_factor, df=3) / scale_factor
# # ax2.plot(x, t_scaled, 'r--', label=f't(3)缩放至方差=1 峰度=∞')
# # ax2.set_title('缩放至相同方差：t分布尖峰厚尾')
# # ax2.legend()
# # ax2.grid(alpha=0.3)
# #
# # plt.tight_layout()
# # plt.show()
# #
#
# # 分析方差贡献
# import numpy as np
# from scipy.stats import t
# import matplotlib.pyplot as plt
#
# # 分析方差贡献
# import numpy as np
# from scipy.stats import t
# import matplotlib.pyplot as plt
#
# # def calculate_variance_decomposition(df=3, threshold=2):
# #     """计算t分布方差在中心和尾部的贡献"""
# #     # 创建足够密集的采样点
# #     x = np.linspace(-30, 30, 50000)
# #     pdf = t.pdf(x, df=df)
# #
# #     # 分割中心区域和尾部
# #     center_mask = np.abs(x) <= threshold
# #     tail_mask = np.abs(x) > threshold
# #
# #     # 计算x²f(x)的积分
# #     def integrate_region(mask):
# #         # 方法1: 使用numpy.trapz（推荐）
# #         try:
# #             return np.trapz(x[mask] ** 2 * pdf[mask], x[mask])
# #         except AttributeError:
# #             # 方法2: 手动实现梯形积分
# #             x_region = x[mask]
# #             y_region = x_region ** 2 * pdf[mask]
# #             return np.sum(0.5 * (y_region[1:] + y_region[:-1]) * np.diff(x_region))
# #
# #     var_center = integrate_region(center_mask)
# #     var_tail = integrate_region(tail_mask)
# #     total_var = var_center + var_tail
# #
# #     return var_center, var_tail, total_var
# #
# #
# # # 计算并显示结果
# # df = 3
# # var_center, var_tail, total_var = calculate_variance_decomposition(df)
# #
# # print("=" * 50)
# # print(f"t分布自由度 ν = {df} 的方差分解")
# # print("=" * 50)
# # print(f"理论方差值: {df / (df - 2):.4f}")
# # print(f"数值计算总方差: {total_var:.4f}")
# # print(f"中心区域(|x|≤2)方差贡献: {var_center:.4f} ({var_center / total_var * 100:.1f}%)")
# # print(f"尾部区域(|x|>2)方差贡献: {var_tail:.4f} ({var_tail / total_var * 100:.1f}%)")
# # print("\n结论：t分布的方差主要由尾部贡献！")
# #
# # # 可视化
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# #
# # # 左图：PDF和区域划分
# # x_plot = np.linspace(-6, 6, 1000)
# # pdf_plot = t.pdf(x_plot, df=df)
# # ax1.plot(x_plot, pdf_plot, 'b-', linewidth=2, label=f't分布 (ν={df})')
# # ax1.fill_between(x_plot[x_plot <= 2], 0, pdf_plot[x_plot <= 2],
# #                  alpha=0.3, color='green', label='中心区域')
# # ax1.fill_between(x_plot[x_plot > 2], 0, pdf_plot[x_plot > 2],
# #                  alpha=0.3, color='red', label='右尾部')
# # ax1.fill_between(x_plot[x_plot < -2], 0, pdf_plot[x_plot < -2],
# #                  alpha=0.3, color='red', label='左尾部')
# # ax1.set_xlabel('x')
# # ax1.set_ylabel('概率密度 f(x)')
# # ax1.set_title(f't分布(ν={df})的概率密度函数')
# # ax1.legend()
# # ax1.grid(True, alpha=0.3)
# #
# # # 右图：x²f(x)的函数
# # ax2.plot(x_plot, x_plot ** 2 * pdf_plot, 'r-', linewidth=2, label='x²f(x)')
# # ax2.fill_between(x_plot[x_plot <= 2], 0, x_plot[x_plot <= 2] ** 2 * pdf_plot[x_plot <= 2],
# #                  alpha=0.3, color='green', label=f'中心贡献: {var_center / total_var * 100:.1f}%')
# # ax2.fill_between(x_plot[x_plot > 2], 0, x_plot[x_plot > 2] ** 2 * pdf_plot[x_plot > 2],
# #                  alpha=0.3, color='red', label=f'右尾贡献')
# # ax2.fill_between(x_plot[x_plot < -2], 0, x_plot[x_plot < -2] ** 2 * pdf_plot[x_plot < -2],
# #                  alpha=0.3, color='red', label=f'左尾贡献: {var_tail / total_var * 100:.1f}%')
# # ax2.set_xlabel('x')
# # ax2.set_ylabel('x²f(x)')
# # ax2.set_title(f'方差贡献函数 (总方差={total_var:.2f})')
# # ax2.legend()
# # ax2.grid(True, alpha=0.3)
# #
# # plt.tight_layout()
# # plt.show()
#
#
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from scipy.stats import kurtosis, skew
# #
# # # 假设 humidity 是你的湿度序列（已过滤错误）
# # print("Skewness:", skew(humidity))
# # print("Kurtosis (Fisher=False):", kurtosis(humidity, fisher=False))  # 正态=3
# #
# #
# # plt.figure(figsize=(10, 4))
# # sns.histplot(humidity, kde=True, bins=50)
# # plt.title("Humidity Distribution")
# # plt.xlabel("Relative Humidity (%)")
# # plt.show()
#
# #
# # import numpy as np
# # from sklearn.preprocessing import QuantileTransformer, PowerTransformer
# #
# # # 创建两个数据集，有相同的排名但不同的数值
# # data1 = np.array([10, 20, 30, 40, 10000])  # 有异常值
# # data2 = np.array([10, 20, 30, 40, 50])     # 无异常值
# #
# # # 分别拟合
# # qt1 = QuantileTransformer(output_distribution='uniform', random_state=42)
# # qt2 = QuantileTransformer(output_distribution='uniform', random_state=42)
# #
# # # 分别变换
# # transformed1 = qt1.fit_transform(data1.reshape(-1, 1))
# # transformed2 = qt2.fit_transform(data2.reshape(-1, 1))
# #
# # print("数据1变换结果:", transformed1.flatten())
# # print("数据2变换结果:", transformed2.flatten())
# # print("\n两个数据集变换后完全一样！")
# #
# # # 现在逆变换
# # inverse1 = qt1.inverse_transform(transformed1)
# # inverse2 = qt2.inverse_transform(transformed2)
# #
# # print("\n逆变换回原始数据:")
# # print("数据1逆变换:", inverse1.flatten())
# # print("数据2逆变换:", inverse2.flatten())
# # print("原始数据1:", data1)
# # print("原始数据2:", data2)
# #
# # # 继续上面的例子，现在用数据1训练的QuantileTransformer处理新数据
# # new_data = np.array([15, 25, 35, 45, 60])  # 与data1类似但没有极端值
# #
# # # 用qt1（基于data1训练）变换新数据
# # new_transformed = qt1.transform(new_data.reshape(-1, 1))
# # print("新数据变换结果:", new_transformed.flatten())
# #
# # # 逆变换
# # new_inverse = qt1.inverse_transform(new_transformed)
# # print("新数据逆变换:", new_inverse.flatten())
# # print("原始新数据:", new_data)
# #
# # print("\n问题：")
# # print("新数据中没有10000这样的极端值")
# # print("但逆变换后的值是基于训练数据分布的，会'回忆'起10000的存在")
# # print("导致逆变换后的值可能被错误地拉伸")
# #
# # import numpy as np
# # from sklearn.preprocessing import QuantileTransformer
# #
# # # 数据1：有异常值
# # data1 = np.array([10, 20, 30, 40, 10000])  # 有极端值10000
# #
# # # 数据2：无异常值
# # data2 = np.array([10, 20, 30, 40, 50])     # 无极端值
#
# # # 创建并拟合两个不同的QuantileTransformer
# # qt1 = QuantileTransformer(output_distribution='uniform', random_state=42)
# # qt2 = QuantileTransformer(output_distribution='uniform', random_state=42)
# #
# # # 分别拟合
# # qt1.fit(data1.reshape(-1, 1))
# # qt2.fit(data2.reshape(-1, 1))
# #
# # # 查看训练后的分位数映射
# # print("qt1的5个分位数（0.0, 0.25, 0.5, 0.75, 1.0）对应的原始值:")
# # print(qt1.quantiles_[:, 0])
# # print("\nqt2的5个分位数（0.0, 0.25, 0.5, 0.75, 1.0）对应的原始值:")
# # print(qt2.quantiles_[:, 0])
# #
# # # 关键理解：每个分位点对应的原始值是不同的！
# # print("\n关键差异：")
# # print(f"qt1的分位数0.5对应: {qt1.quantiles_[2, 0]} (数据1的中位数)")
# # print(f"qt2的分位数0.5对应: {qt2.quantiles_[2, 0]} (数据2的中位数)")
# #
# # # 用qt1（基于有异常值的数据训练）来处理新数据
# # new_data = np.array([15, 25, 35, 45, 60])  # 新数据，无极端值
# #
# # # 变换
# # new_transformed = qt1.transform(new_data.reshape(-1, 1)).flatten()
# # print("新数据变换结果（使用qt1）：", new_transformed)
# #
# # # 逆变换
# # new_inverse = qt1.inverse_transform(new_transformed.reshape(-1, 1)).flatten()
# # print("新数据逆变换结果：", new_inverse)
# # print("原始新数据：", new_data)
# # print()
# #
# # # 对比：如果新数据来自与训练数据相同的分布
# # new_data_same_dist = np.array([15, 25, 35, 45, 10000])  # 与data1分布相同
# # new_transformed_same = qt1.transform(new_data_same_dist.reshape(-1, 1)).flatten()
# # new_inverse_same = qt1.inverse_transform(new_transformed_same.reshape(-1, 1)).flatten()
# #
# # print("同分布新数据变换结果：", new_transformed_same)
# # print("同分布新数据逆变换结果：", new_inverse_same)
# # print("原始同分布数据：", new_data_same_dist)
#
#
# # # 创建一个更直观的例子
# # np.random.seed(42)
# #
# #
# # # 训练数据：收入数据（单位：千元）
# # train_income = np.array([30, 40, 50, 60, 70, 80, 90, 100, 200, 500])
# #
# # qt_income = QuantileTransformer(output_distribution='uniform', random_state=42)
# # qt_income.fit(train_income.reshape(-1, 1))
# #
# # # 新数据1：另一个公司的收入数据（分布类似但数值不同）
# # new_income1 = np.array([25, 35, 45, 55, 65, 75, 85, 95, 150, 300])
# #
# # # 新数据2：与训练数据完全不同的分布
# # new_income2 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
# #
# # transformed1 = qt_income.transform(new_income1.reshape(-1, 1)).flatten()
# # inverse1 = qt_income.inverse_transform(transformed1.reshape(-1, 1)).flatten()
# # print("变换1：", transformed1)
# # print("逆变换1：", inverse1) #  [ 30.  35.  45.  55.  65.  75.  85.  95. 150. 300.]
# #
# # transformed2 = qt_income.transform(new_income2.reshape(-1, 1)).flatten()
# # inverse2 = qt_income.inverse_transform(transformed2.reshape(-1, 1)).flatten()
# # print("逆变换2：", inverse2) # [100. 200. 300. 400. 500. 500. 500. 500. 500. 500.]
#
#
# # 假设有以下特征分类
# # boundary_features = ['rh']  # 边界堆积特征
# # skewed_features = ['T', 'pressure']  # 偏态特征
# # other_features = ['wind_speed', 'cloud_cover']  # 其他特征
# #
# # # 构建每个特征的管道
# # boundary_pipe = Pipeline([
# #     ('quantile', QuantileTransformer(output_distribution='normal', random_state=42))
# # ])
# # # 注意：这里没有标准化，因为QuantileTransformer已经输出正态分布
# #
# # skewed_pipe = Pipeline([
# #     ('power', PowerTransformer(standardize=False)),
# #     ('scaler', RobustScaler())
# # ])
# #
# # other_pipe = Pipeline([
# #     ('scaler', RobustScaler())
# # ])
#
# # # 组合成ColumnTransformer
# # preprocessor = ColumnTransformer([
# #     ('boundary', boundary_pipe, boundary_features),
# #     ('skewed', skewed_pipe, skewed_features),
# #     ('other', other_pipe, other_features)
# # ])
# #
# # # 在训练数据上拟合
# # preprocessor.fit(X_train)
# #
# # # 变换训练数据
# # X_train_transformed = preprocessor.transform(X_train)
# #
# # # 变换新数据（预测时）
# # X_new_transformed = preprocessor.transform(X_new)
#
# #
# #
# # import numpy as np
# # from sklearn.base import BaseEstimator, TransformerMixin
# # from sklearn.preprocessing import FunctionTransformer
# #
# # # 最简单的方式：使用FunctionTransformer
# # asinh_transformer = FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh)
# #
# # # 使用示例
# # wv_x = np.array([0, 0.1, 0.3, -0.2, 5.0, -8.0, 0.001, 20.0, -15.0])
# # wv_x_transformed = asinh_transformer.fit_transform(wv_x.reshape(-1, 1)).flatten()
# #
# # print("原始数据:")
# # print(wv_x)
# # print("\nasinh变换后:")
# # print(wv_x_transformed)
# # print("\n关键观察:")
# # print(f"0 -> {np.arcsinh(0):.6f}")
# # print(f"0.1 -> {np.arcsinh(0.1):.6f}")
# # print(f"5.0 -> {np.arcsinh(5.0):.6f}")
# # print(f"20.0 -> {np.arcsinh(20.0):.6f}")
# # print(f"-0.2 -> {np.arcsinh(-0.2):.6f}")
# # print(f"-8.0 -> {np.arcsinh(-8.0):.6f}")
# #
# #
# # class AsinhTransformer(BaseEstimator, TransformerMixin):
# #     """自定义的反双曲正弦变换器"""
# #
# #     def __init__(self, scale_factor=1.0):
# #         """
# #         参数:
# #         scale_factor: 缩放因子，可以调整变换的敏感性
# #         asinh(x/scale_factor)会先缩放数据
# #         """
# #         self.scale_factor = scale_factor
# #
# #     def fit(self, X, y=None):
# #         # asinh变换无状态，直接返回self
# #         return self
# #
# #     def transform(self, X):
# #         X = X.copy()
# #         return np.arcsinh(X / self.scale_factor)
# #
# #     def inverse_transform(self, X):
# #         return np.sinh(X) * self.scale_factor
# #
# #     def set_params(self, **params):
# #         for key, value in params.items():
# #             setattr(self, key, value)
# #         return self
# #
# #
# # # 使用示例
# # transformer = AsinhTransformer(scale_factor=1.0)
# #
# # # 模拟风速分量数据
# # np.random.seed(42)
# # # 生成大量接近0的小值和少量大值
# # small_winds = np.random.uniform(-1, 1, 950)  # 95%小风
# # strong_winds = np.random.uniform(-20, 20, 50)  # 5%大风
# # wv_x_sample = np.concatenate([small_winds, strong_winds])
# # np.random.shuffle(wv_x_sample)
# #
# # print("数据统计:")
# # print(f"样本数: {len(wv_x_sample)}")
# # print(f"均值: {wv_x_sample.mean():.4f}")
# # print(f"标准差: {wv_x_sample.std():.4f}")
# # print(f"最小值: {wv_x_sample.min():.4f}")
# # print(f"最大值: {wv_x_sample.max():.4f}")
# # print(f"|值|<0.5的比例: {np.mean(np.abs(wv_x_sample) < 0.5):.2%}")
# # print(f"|值|<1.0的比例: {np.mean(np.abs(wv_x_sample) < 1.0):.2%}")
# #
# # # 应用变换
# # wv_x_transformed = transformer.transform(wv_x_sample)
# #
# # print("\nasinh变换后统计:")
# # print(f"均值: {wv_x_transformed.mean():.4f}")
# # print(f"标准差: {wv_x_transformed.std():.4f}")
# # print(f"最小值: {wv_x_transformed.min():.4f}")
# # print(f"最大值: {wv_x_transformed.max():.4f}")
# #
# # # 对比关键值的变化
# # test_values = np.array([0, 0.1, 0.5, 1.0, 5.0, 10.0, -0.1, -0.5, -1.0, -5.0, -10.0])
# # print("\n关键值变换对比:")
# # print("原始值 -> asinh变换值")
# # for val in test_values:
# #     print(f"{val:6.1f} -> {np.arcsinh(val):8.4f}")
# #
# # arr_3d = np.zeros((2, 3, 4))
# # """
# # [[[0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]]
# #
# #  [[0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]]]
# # """
# # print(arr_3d)
# # slice_2d = np.array([[1, 2, 3, 4],
# #                      [5, 6, 7, 8],
# #                      [9, 10, 11, 12]])
# # arr_3d[:, :, :] = slice_2d[np.newaxis, :, :]
# # # arr_3d[...] = slice_2d
# # arr_3d[:, :, :] =slice_2d
# # print('\n',arr_3d)
#
#
# # print(np.broadcast_shapes((2,3,4), (3,4)))
# #
# # res[:, col_index] = data_2d.flatten()
# # res[:, :] = data_2d.flatten()
# import numpy as np
#
# #
# # # 示例验证
# # res = np.zeros((3, 4))  # 形状 (3, 4)
# # data = np.arange(3)     # 形状 (3,)
# #
# # res[:, :] = data  # 错误 ❌(3,) 补1到二维应该是 (1, 3)，而不是 (3, 1)
# # # (3,)，values.它本身 没有明确的行或列方，两数组维度不一样的时候，numPy 会在较小的数组 前面补1，使其维度对齐
# # # 当是（1，3）的时候，与（3，4）对不齐 报错
# # res[:,:] = data[:, np.newaxis] # 正确 补齐成(3,1)
# # res[:,:] = data[:, None] # 正确 补齐成(3,1)
# # print(res)
# # """
# # 输出：
# # [[0. 0. 0. 0.]
# #  [1. 1. 1. 1.]
# #  [2. 2. 2. 2.]]
# # """
# # arr = np.zeros((3,4))
# # arr[:,0] = [1,2,3] # 正确 (直接赋值，不需要广播)
# # arr[:,0] = np.array([[1,2,3]]) # (1，3) 正确
# # arr[:,0] = np.array([[1],[2],[3]]) # (3,1) ❌
# # print(arr) # 应用PowerTransformer多列转换
# """
# [[1. 0. 0. 0.]
#  [2. 0. 0. 0.]
#  [3. 0. 0. 0.]]
#  """
# # 关键区别
# # 例1：res[:, :] 是一个完整的二维数组视图，形状是 (3, 4)
# # 例2：arr[:, 0] 是一个一维列切片，形状是 (3,)
#
# # 将多个列表"并排"组合
# names = ['Alice', 'Bob', 'Charlie']
# ages = [25, 30, 35]
# scores = [85, 92, 78]
#
# paired = list(zip(names, ages, scores))
# # [('Alice', 25, 85), ('Bob', 30, 92), ('Charlie', 35, 78)]
#
# # 按某个列表排序，同时保持其他列表对应关系
# names = ['Charlie', 'Alice', 'Bob']
# scores = [78, 85, 92]
#
# # 按分数排序
# sorted_pairs = sorted(zip(scores, names))
# # [(78, 'Charlie'), (85, 'Alice'), (92, 'Bob')]
#
# # 按名字排序
# sorted_by_name = sorted(zip(names, scores))
#
#
# #  [('Alice', 85), ('Bob', 92), ('Charlie', 78)]
#
#
# # def calculate_total(price, quantity, tax_rate):
# #     return price * quantity * (1 + tax_rate)
# #
# #
# # prices = [10, 20, 30]
# # quantities = [2, 3, 1]
# # tax_rates = [0.1, 0.15, 0.2]
# #
# # totals = list(map(calculate_total, prices, quantities, tax_rates))  # map(func, *iterables)
# # # Make an iterator that computes the function using arguments from each of the iterables
# # a = [calculate_total(*args) for args in zip(prices, quantities, tax_rates)]
# # print(a)
# # print(totals)
#
# # 筛选及格的人
# # names = ['Alice', 'Bob', 'Charlie']
# # scores = [85, 42, 92]
# #
# # a = [(name, scores) for name, scores in zip(names, scores) if scores >= 60]
# # b = list(filter(lambda x: x[1] >= 60, zip(names, scores)))  # filter(func, *iterables)
# # print(b)
# #
# #
# # from collections import defaultdict, Counter
# #
# # # 分组统计(原格式 元组）
# # data = [('apple', 'fruit'), ('carrot', 'vegetable'),
# #         ('banana', 'fruit'), ('potato', 'vegetable')]
# #
# # grouped = defaultdict(list)
# # for item, category in data: # 直接解包元组
# #     grouped[category].append(item)
# #     # defaultdict(<class 'list'>, {'fruit': ['apple', 'banana'], 'vegetable': ['carrot', 'potato']})
# #
# # # 计数
# # counts = Counter(grouped.get('fruit',None))
# # # Counter({'apple': 1, 'banana': 1})
# import pandas as pd
#
# import numpy as np
#
# # 示例数据
# X = np.array([
#     [1.0, 2.0, np.nan],   # 第0行：有NaN
#     [4.0, 5.0, 6.0],      # 第1行：没有NaN
#     [np.nan, 8.0, 9.0],   # 第2行：有NaN
#     [10.0, 11.0, 12.0]    # 第3行：没有NaN
# ])
#
# # 步骤1: np.isnan(X_np) - 找出所有NaN位置
# nan_mask = np.isnan(X)
# """
# array([[False, False,  True],  # 第0行，第2列是NaN
#        [False, False, False],  # 第1行，没有NaN
#        [ True, False, False],  # 第2行，第0列是NaN
#        [False, False, False]]) # 第3行，没有NaN
# """
#
# # 步骤2: np.any(..., axis=1) - 检查每行是否有至少一个True
# row_mask = np.any(nan_mask, axis=1)
# """
# array([ True,  False,  True,  False])
# # 解释：
# # 第0行：有True → True
# # 第1行：全False → False
# # 第2行：有True → True
# # 第3行：全False → False
# """
#
# # 步骤3: 用~取反，选择没有NaN的行
# X_clean = X[~row_mask]
# """
# array([[ 4.,  5.,  6.],
#        [10., 11., 12.]])
# """
#
#

import os

checkpoint_path = '/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm2#_20260208_154953'

print("遍历结果:")
for root, dirs, files in os.walk(checkpoint_path):
    print(f"\n当前目录: {root}")
    print(f"子目录: {dirs}")
    print(f"文件: {files}")

    # 如果需要处理 keras 文件
    for file in files:
        if file.endswith('.keras'):
            full_path = os.path.join(root, file)
            print(f"找到Keras文件: {full_path}")

# class ForceLRCallback(tf.keras.callbacks.Callback):
#     def __init__(self, start_epoch=33):
#         super().__init__()
#         self.start_epoch = start_epoch
#         self.call_count = 0  # 记录调用次数（陷阱：第一次调用传递非0，后续调用循环才是0，1，2）
#
#     def on_epoch_begin(self, epoch, logs=None):
#         self.call_count += 1
#
#         if self.call_count == 1:
#             # 第一次调用：初始化
#             actual_epoch = self.start_epoch
#         else:
#             # 后续调用：epoch是相对值！比如：0,1,2...对应实际33,34,35...
#             actual_epoch = self.start_epoch + self.call_count
#
#         if actual_epoch <= 38:
#             target_lr = 2.5e-05
#         elif actual_epoch <= 43:
#             target_lr = 2.0e-05
#         elif actual_epoch <= 48:
#             target_lr = 1.2e-05
#         else:
#             target_lr = 8e-05
#
#         # 关键，修改现有优化器的学习率
#         try:
#             self.model.optimizer.learning_rate.assign(target_lr)
#             logger.debug(f"Epoch_{actual_epoch}: 强制设置LR = {target_lr:.2e}")
#
#         except Exception as e:
#             logger.debug(f"无法修改学习率: {e}")
#             old = self.model.optimizer
#
#             self.model.optimizer = tf.keras.optimizers.Adam(
#                 learning_rate=target_lr,
#                 beta_1=getattr(old, 'beta_1', 0.9),
#                 beta_2=getattr(old, 'beta_2', 0.999),
#                 epsilon=getattr(old, 'epsilon', 1e-7)
#             )
#             logger.debug(f"创建新优化器，但复制了超参数")


# import hashlib
# import numpy as np
#
# def get_batch_signature(dataset, batch_index=0):
#     """
#     获取指定 batch 的数据签名，自动探测数据结构
#     """
#     for i, batch in enumerate(dataset):
#         if i == batch_index:
#
#             # 尝试获取实际的数据张量
#             def extract_tensor(data):
#                 """递归提取第一个遇到的张量"""
#                 if hasattr(data, 'numpy'):  # 是 Tensor
#                     return data
#                 elif isinstance(data, (tuple, list)) and len(data) > 0:
#                     return extract_tensor(data[0])
#                 elif isinstance(data, dict):
#                     first_key = list(data.keys())[0]
#                     return extract_tensor(data[first_key])
#                 else:
#                     return None
#
#             # 提取第一个张量
#             tensor = extract_tensor(batch)
#
#             if tensor is not None:
#                 x_bytes = tensor.numpy().tobytes()
#                 hash_value = hashlib.md5(x_bytes).hexdigest()
#
#                 stats = {
#                     'shape': list(tensor.shape),
#                     'mean': float(np.mean(tensor.numpy())),
#                     'std': float(np.std(tensor.numpy())),
#                     'min': float(np.min(tensor.numpy())),
#                     'max': float(np.max(tensor.numpy())),
#                     'hash': hash_value,
#                 }
#                 return stats
#             else:
#                 print("无法找到张量数据")
#                 return None
#     return None
#
# # 使用
# for batch_idx in range(5):
#     signature1 = get_batch_signature(trainset, batch_idx)
#     print(f"trainsetBatch {batch_idx} hash: {signature1['hash'][:8]}...")
#     signature2 = get_batch_signature(valset, batch_idx)
#     print(f"trainsetBatch {batch_idx} hash: {signature2['hash'][:8]}...")
#
# checkpoint_dir='/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm2*_20260304_094300/tf_checkpoints_stage0/epoch_22'
#
# os.listdir(checkpoint_dir)

# def _compile_for_prediction_model(self, model):  # 同一Python进程中直接获取实例。独立的演化路径
#     """为预测模型重新编译 多输出会折叠metrics会折叠"""
#
#     # 获取实际输出数量
#     num_outputs = len(model.outputs)
#     logger.debug(f"模型有 {num_outputs} 个输出")
#
#     # 获取输出层名称（使用模型输出层名称，不是张量名称）
#     output_names = []
#     for output in model.outputs:
#         for layer in model.layers:
#             if hasattr(layer, 'output') and layer.output is output:
#                 output_names.append(layer.name)
#                 break
#     logger.debug(f"输出层名称：{output_names}")
#
#     # 构建字典配置
#     # 使用统一的配置管理器
#     loss_config = ModelConfigManager.get_loss_config(self.model_config)
#     metrics_config = ModelConfigManager.get_metrics_config(self.model_config)
#     loss_weights_config = ModelConfigManager.get_loss_weights_config(self.model_config)
#
#     logger.debug(f"loss_config: {loss_config}")
#     logger.debug(f"metrics_config: {metrics_config}")
#     logger.debug(f"loss_weights_config:{loss_weights_config}")
#
#     # 获取优化器
#     if hasattr(self, 'training_model_') and hasattr(self.training_model_, 'optimizer'):
#         optimizer = self.training_model_.optimizer  # 可以用实例，load可以用配置
#     else:
#         learning_rate = self.model_config.get('learning_rate', 0.001)
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
#     # 单输出或者多输出都可以使用字典，但是要保证输出层名字正确
#     logger.debug("=== 编译前检查 ===")
#     logger.debug(f"输出层: {output_names}")
#     logger.debug(f"loss_config: {loss_config}")
#     logger.debug(f"metrics_config: {metrics_config}")
#     logger.debug(f"loss_config类型: {type(loss_config)}")
#     logger.debug(f"metrics_config类型: {type(metrics_config)}")
#
#     model.compile(
#         optimizer=optimizer,
#         loss=loss_config,  # 字典 键是输出层名
#         loss_weights=loss_weights_config,
#         metrics=metrics_config
#     )
#
#     logger.debug("编译完成，验证metrics配置...")
#
#     if len(model.metrics) >= 2:
#         compile_metrics = model.metrics[1]
#         if hasattr(compile_metrics, '_user_metrics'):
#             actual_metrics = compile_metrics._user_metrics
#             logger.debug(f"实际编译的metrics配置: {actual_metrics}")
#             logger.debug(f"期望的metrics配置: {metrics_config}")
#
#     return model

import math
import tensorflow as tf
# def optimal_cosine_annealing_with_start( epoch,warmup_epochs,total_epochs,initial_lr,min_lr,warmup_power,start_epoch=23):
#     """支持从中间epoch开始的余弦退火"""
#
#     # 调整epoch：减去开始epoch
#     adjusted_epoch = epoch - start_epoch + 1
#
#     # 如果adjusted_epoch已经在warmup之后
#     if adjusted_epoch >= warmup_epochs:
#         # 直接进入余弦衰减阶段
#         decay_epoch = adjusted_epoch - (warmup_epochs - 1)
#         decay_total = total_epochs - warmup_epochs + 1
#
#         # 余弦衰减
#         progress = decay_epoch / decay_total
#         cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
#
#         return min_lr + (initial_lr - min_lr) * cosine_decay
#     else:
#         # 还在warmup阶段（不太可能）
#         progress = adjusted_epoch / warmup_epochs
#         return min_lr + (initial_lr - min_lr) * (progress ** warmup_power)
#
# for epoch in range(23,25):
#     # warmup_epochs =1 代表热身0轮（不用热身）第23轮就开始衰减
#     res = optimal_cosine_annealing_with_start(epoch,start_epoch=23,warmup_epochs=1,total_epochs=20,initial_lr=0.00035,min_lr=1e-6,warmup_power=2)
#     print(res)
#
# for epoch in range(23,50):
#     # 需要热身 warmup_epochs =2，代表热身1轮（即epoch=23 热身，之后衰减）
#     res2 = optimal_cosine_annealing_with_start(epoch,start_epoch=23,warmup_epochs=2,total_epochs=20,initial_lr=0.00035,min_lr=1e-6,warmup_power=2)
#     print(res2)
# class CosineAnnealingWarmRestarts(tf.keras.callbacks.Callback):
#     def __init__(self, initial_lr=0.00035, min_lr=1e-6, total_epochs=30, warmup_epochs=5,
#                  warmup_power=2.0, restart_epochs: list = None):
#         super().__init__()
#         self.initial_lr = initial_lr
#         self.min_lr = min_lr
#         self.total_epochs = total_epochs
#         self.warmup_epochs = warmup_epochs
#         self.warmup_power = warmup_power
#         self.restart_epochs = restart_epochs or []
#         """
#         参数:
#         - initial_lr: 初始学习率 (0.00035)  -> 顶
#         - min_lr: 最小学习率 (1e-6) ->  脚
#         - total_epochs: 1周期的总epoch数
#         - warmup_epochs: warmup阶段epoch数（先小学习率"热身"，再大学习率训练）
#         - warmup_power: warmup曲线形状 (1=线性（直线）, 2=二次（曲线）)
#         - restart_epochs: 重启点列表，如[15, 25]表示在第15、25个epoch重启
#         """
#
#     def optimal_cosine_annealing(self, epoch):
#         """
#         - epoch: 当前epoch
#         """
#         # 处理重启逻辑
#         if self.restart_epochs and len(self.restart_epochs) > 0:
#             restart_epochs = sorted(self.restart_epochs)
#             current_cycle_start = 0
#             cycle_length = self.total_epochs
#
#             for i in range(len(restart_epochs)):
#                 restart_epoch = restart_epochs[i]
#                 if epoch >= restart_epoch:
#                     current_cycle_start = restart_epoch
#
#                     # 计算当前周期的长度
#                     if i + 1 < len(restart_epochs):
#                         next_restart = restart_epochs[i + 1]
#                         cycle_length = next_restart - restart_epoch
#                     else:
#                         cycle_length = self.total_epochs - restart_epoch
#                 else:
#                     # 处理第一个周期（0到第一个重启点）的情况
#                     if i == 0:
#                         cycle_length = restart_epoch - 0
#                     break
#
#             epoch_in_cycle = epoch - current_cycle_start
#             effective_total = cycle_length
#         else:
#             epoch_in_cycle = epoch
#             effective_total = self.total_epochs
#
#         # 还在Warmup阶段
#         if epoch_in_cycle+1 < self.warmup_epochs:
#             warmup_progress = (epoch_in_cycle + 1) / self.warmup_epochs # 是当前周期内的相对位置 归一化到[0, 1]范围
#             warmup_factor = warmup_progress ** self.warmup_power
#             return self.min_lr + (self.initial_lr - self.min_lr) * warmup_factor  # 确保学习率始终大于最小
#
#         if effective_total <= self.warmup_epochs:
#             return self.min_lr
#
#         # 余弦退火阶段
#         adjusted_epoch = (epoch_in_cycle+1) - (self.warmup_epochs-1)
#         adjusted_total = effective_total - (self.warmup_epochs-1)
#
#         # 确保不除零
#         if adjusted_total <= 0:
#             return self.min_lr
#
#         progress = adjusted_epoch / adjusted_total
#
#         cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
#
#         return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
#
# for epoch in range(20) :
#     cosine_callback = CosineAnnealingWarmRestarts(
#                 initial_lr=0.00039,
#                 min_lr=1e-5,
#                 total_epochs=20,  # 1周期总轮数
#                 warmup_epochs=1,  # 4代表3轮热身 / 如果需要早停 耐心值至少是warmup_epochs的3-5倍
#                 warmup_power=2.0,
#                 restart_epochs=[10])
#     # 不需要热身，直接衰减
#     res=cosine_callback.optimal_cosine_annealing(epoch)
#     print(f'epoch:{epoch}:{res}')
#
# for epoch in range(21) :
#     cosine_callback = CosineAnnealingWarmRestarts(
#                 initial_lr=0.00039,
#                 min_lr=1e-5,
#                 total_epochs=20,  # 1周期总轮数
#                 warmup_epochs=2,  # 4代表3轮热身 / 如果需要早停 耐心值至少是warmup_epochs的3-5倍
#                 warmup_power=2.0,
#                 restart_epochs=[10])
#     # 需要热身1轮后衰减
#     res2=cosine_callback.optimal_cosine_annealing(epoch)
#     print(f'epoch:{epoch}:{res2}')
#


def cal_metric(min_delta,current_val_loss,current_loss):
    best_val_loss =0.060108
    best_loss =0.05865

    if current_val_loss < best_val_loss - min_delta:
        current_gap_abs = abs(current_val_loss - current_loss)
        best_gap_abs = abs(best_val_loss - best_loss)   # 需要记录最佳时的训练损失
        if current_gap_abs <= best_gap_abs * 1.1:        # 允许绝对差距小幅增大
            best_val = current_val
            best_train_loss = train_loss
            save_model()
            return '可更新'
        else:
            return '不能更新'


min_delta = 1e-6


current_val_loss =0.060082
current_loss = 0.05845
print(cal_metric(min_delta,current_val_loss,current_loss))

# 参数设置
min_delta = 1e-6  # 建议设大一点，忽略噪声
gap_tolerance_ratio = 1.1  # 允许 Gap 增大 10%
min_gap_threshold = 0.001   # 防止过程中的 Gap 过小： 0*1.1=0 导致的误杀 (根据量级调整) ，同时也不能太小

if current_val_loss < best_val_loss - min_delta:
    current_gap = abs(current_val_loss - current_loss)
    best_gap = abs(best_val_loss - best_loss)

    allowed_gap = max(best_gap * gap_tolerance_ratio, min_gap_threshold)

    if current_gap <= allowed_gap:
        best_val_loss = current_val_loss
        best_loss = current_loss
        best_gap_recorded = current_gap
        save_model()
        print(
            f"模型已保存 (Epoch {epoch}): Val Loss 显著下降且 Gap ({current_gap:.5f}) 在允许范围 ({allowed_gap:.5f}) 内")
    else:
        # ⚠️ 警惕：Loss 降了，但过拟合加剧太多，放弃保存
        print(
            f"跳过保存 (Epoch {epoch}): Val Loss 虽下降，但 Gap ({current_gap:.5f}) 超出允许范围 ({allowed_gap:.5f})，疑似过拟合。")
else:
    # Loss 没怎么降，直接跳过
    pass