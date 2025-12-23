# 创建并编译一个模型
original_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

original_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 保存为 .keras
original_model.save('test_model.keras')

# 加载模型
loaded_model = tf.keras.models.load_model('test_model.keras')

# 检查编译状态
print("原始模型:")
print(f"  优化器: {original_model.optimizer}")
print(f"  Loss: {original_model.loss}")
print(f"  Metrics: {original_model.metrics}")

print("\n加载的模型:")
print(f"  优化器: {loaded_model.optimizer}")
print(f"  Loss: {loaded_model.loss}")
print(f"  Metrics: {loaded_model.metrics}")

# 检查是否可以直接使用
print(f"\n是否可以直接predict? {'✅' if hasattr(loaded_model, 'predict') else '❌'}")
print(f"是否可以直接evaluate? {'✅' if hasattr(loaded_model, 'evaluate') else '❌'}")
print(f"是否可以直接compile? {'✅' if hasattr(loaded_model, 'compile') else '❌'}")

# 验证 metrics 配置
print(f"\n模型metrics列表: {loaded_model.metrics}")

# def get_deployment_model(self):
#     """获取部署用的SavedModel路径"""
#
#     if not hasattr(self, 'best_checkpoint'):
#         raise ValueError('未找到最佳模型检查点')
#
#     savedmodel_dir = os.path.join(self.best_checkpoint, 'saved_model')
#
#     if not os.path.exists(savedmodel_dir):
#         raise FileNotFoundError(
#             f"找不到SavedModel目录: {savedmodel_dir}\n"
#             "请在训练回调中确保同时保存了SavedModel格式"
#         )
#
#     # 验证SavedModel格式
#     if not os.path.exists(os.path.join(savedmodel_dir, 'saved_model.pb')):
#         raise ValueError(f"不是有效的SavedModel格式: {savedmodel_dir}")
#
#     print(f"✅ 部署模型位置: {savedmodel_dir}")
#     return savedmodel_dir
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


# def save_for_deployment(self, deploy_path):
#     """ 只保存部署格式 - 从训练检查点复制SavedModel Args: deploy_path: 部署目录 """
#     check_is_fitted(self)
#     if not hasattr(self, 'best_checkpoint'):
#         raise ValueError('需要先训练并保存最佳检查点')
#
#     # 1. 保存模型
#     os.makedirs(deploy_path, exist_ok=True)
#
#     source_savedmodel = os.path.join(self.best_checkpoint, 'saved_model')
#     target_savedmodel = os.path.join(deploy_path, 'saved_model')
#
#     if not os.path.exists(source_savedmodel):
#         logger.warning(f"训练检查点中没有SavedModel，创建新的")
#         if not hasattr(self, '_prediction_model'):
#             self._prediction_model = self.reconstruct_model()
#         self._prediction_model.save(target_savedmodel, save_format='tf')  # export()
#     else:
#         if os.path.exists(target_savedmodel):
#             shutil.rmtree(target_savedmodel)
#         shutil.copytree(source_savedmodel, target_savedmodel)
#     logger.info(f"SavedModel部署格式：{target_savedmodel}")
#
#     # 2. 保存预处理流水线
#
#
#     # 保存配置信息
#     deploy_config = {
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
#
#         # # 必需：特征工程配置
#         # 'feature_config': {
#         #     'input_columns': self._get_input_columns(),
#         #     'output_columns': self._get_output_columns(),
#         #     'scalers': self._get_scaler_info(),  # 标准化器信息
#         #     'encoders': self._get_encoder_info(),  # 编码器信息
#         # },
#         # 预处理
#         # 'data_processing': self._get_data_processing_config(),
#         # 'preprocessing': {
#         #     'required_columns': self._get_required_columns(),
#         #     'normalization': self._get_normalization_info(),
#         #
#
#         'deployment_info': {
#             'purpose': 'deployment_only',
#             'version': '1.0',
#             'tensorflow_version': tf.__version__,
#             'save_time': datetime.now().isoformat(),
#             'training_checkpoint': self.best_checkpoint if hasattr(self, 'best_checkpoint') else None
#         }
#
#     }
#
#     with open(os.path.join(deploy_path, 'deploy_config.json'), 'w') as f:
#         json.dump(deploy_config, f, indent=2, default=str)
#
#     logger.info(f"✅ 部署包已保存: {deploy_path}")
#     logger.info(f"   - SavedModel: {target_savedmodel}")
#     logger.info(f"   - 配置: deploy_config.json")
#
#     return deploy_path

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