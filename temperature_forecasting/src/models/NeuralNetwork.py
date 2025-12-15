import os
from typing import Dict, Any

import joblib
from pydantic.v1 import validate_arguments
from pydantic import Field
from sklearn.utils.validation import check_is_fitted
from data.decorator import validate_input
from models.cnn import MultiTasksCnnModel
from models.lstm import SingleTaskLstmModel, MultiTasksLstmModel
from training.training_models import TrainingMultiModel, TrainingSingleModel
from data.windows import EnhancedWindowGenerator
from evaluation.model_evaluation import ModelEvaluation
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import tensorflow as tf
import pandas as pd
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


class TimeSeriesEstimator(BaseEstimator, RegressorMixin, ClassifierMixin):
    @validate_arguments
    def __init__(self,
                 model_config: dict = Field(..., description="必须提供包括模型配置（output_config）在内的、窗口配置。")):
        """
        Parameters:
        -----------
        model_config : dict, optional
            模型配置，用于训练新模型。通过各自模型进行参数验证
        saved_model_path : str, optional
            已保存模型路径，如果提供则直接使用保存的模型
        """
        self.model_config = model_config or {}
        self.best_checkpoint = None

        self.training_model_ = None  # 训练过程中使用的模型（可能包含dropout等）
        self.prediction_model_ = None  # 专门用于预测的最佳模型（已加载最佳权重）
        self.is_fitted_ = False
        self.embedding_info_ = {}
        self.history_ = None
        self.train_window_data = None
        self.val_window_data = None
        self.test_window_data = None
        self.input_cols_ = None  # 处理掉时间列，保证进入模型的所有列是数值

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.weights_dir = f"saved_model/{self.model_config['model_type']}_{timestamp}"

    def fit(self, X: dict, y=None):
        # 写出数据源
        train_datasets = X['train_datasets']
        val_datasets = X['val_datasets']

        # 0.处理时间列
        datetime_cols = train_datasets.select_dtypes(include=['datetime64']).columns
        self.input_cols_ = [col for col in list(train_datasets.columns) if col not in datetime_cols]

        train_datasets_ = train_datasets[self.input_cols_]
        val_datasets_ = val_datasets[self.input_cols_]

        #  构建模型 （神经网络预处理已经返回了模型期望的正确格式，不copy）
        # 1.1 获得embedding_info
        self.embedding_info = EmbeddingConfig._get_embedding_info(train_datasets_,  # 原始DF
                                                                  self.model_config['categorical_columns']
                                                                  )

        # 1.2 处理窗口数据
        self.window = self._create_window_generator(self.embedding_info, self.model_config['output_config'])
        train_window_data = self.window.createMultimodalDataset(train_datasets_)
        val_window_data = self.window.createMultimodalDataset(val_datasets_)

        # 多任务
        if self.model_config['multi_tasks']:

            if self.model_config['model_type'].startswith('multi_lstm'):
                lstm_model_config = {**self.model_config,
                                     'embedding_configs': self.embedding_info}
                lstm_model = MultiTasksLstmModel(lstm_model_config)
                lstm_model_ = lstm_model._build_lstm_model()
                self.training_model_ = lstm_model_

            elif self.model_config['model_type'].startswith('multi_cnn'):
                cnn_model_config = {**self.model_config,  # 解包
                                    'embedding_configs': self.embedding_info}  # 追加
                cnn_model = MultiTasksCnnModel(architecture_type='enhance_parallel', config=cnn_model_config)
                cnn_model_ = cnn_model._build_cnn_model()
                self.training_model_ = cnn_model_

            # 1.4 训练模型
            # 确保目录存在(立即创建)
            os.makedirs(self.weights_dir, exist_ok=True)

            self.history_, best_checkpoint = TrainingMultiModel(model_name=self.model_config['model_type'],
                                                                model=self.training_model_,
                                                                trainset=train_window_data,
                                                                valset=val_window_data,
                                                                verbose=self.model_config['verbose'],
                                                                epochs=self.model_config['epochs'],
                                                                weights_dir=self.weights_dir)
        # 单任务
        else:
            if self.model_config['model_type'].startswith('single_lstm'):
                lstm_model_config = {**self.model_config,
                                     'embedding_configs': self.embedding_info}
                lstm_model = SingleTaskLstmModel(lstm_model_config)
                lstm_model_ = lstm_model._build_lstm_model()
                self.training_model_ = lstm_model_

            os.makedirs(self.weights_dir, exist_ok=True)
            self.history_, best_checkpoint = TrainingSingleModel(model_name=self.model_config['model_type'],
                                                                 model=self.training_model_,
                                                                 trainset=train_window_data,
                                                                 valset=val_window_data,
                                                                 verbose=self.model_config['verbose'],
                                                                 epochs=self.model_config['epochs'],
                                                                 weights_dir=self.weights_dir)

        # 保存最佳检查点路径供后续使用
        self.best_checkpoint = best_checkpoint

        # 训练完成后，创建用于预测的模型
        self._prediction_model = self.reconstruct_model()

        # 1.5 评估模型
        self.evaluate_model(dataset=val_window_data, dataset_type='val')

        self.is_fitted_ = True

        return self

    @validate_input(validate_y=False)
    def predict(self, X):
        check_is_fitted(self)

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_model = X_[self.input_cols_]
        time_column_X_ = X_[self.model_config['time_column']]

        # 1. 处理窗口数据
        predict_window_data = self.window.createMultimodalDataset(X_model)

        # 2. 重构模型
        if self.prediction_model_ is None:
            self._prediction_model = self.reconstruct_model()  # 确保使用最佳权重

        # 3. 模型预测
        predictions = self._prediction_model.predict(predict_window_data)  # 多输入和输出（tuple,dict）->预测结果是list

        # 4. 恢复未使用时间列
        historical_timestamps = time_column_X_.copy()

        last_time = historical_timestamps.iloc[-1]
        steps_ahead = self.model_config['output_width']  # 默认预测步长

        future_timestamps = self._generate_future_timestamps(last_time, self.model_config['output_width'], 'H')

        predictions_ = pd.DataFrame({
            'timestamp': future_timestamps,
            'prediction': predictions.flatten()[:steps_ahead]  # 确保长度匹配
        })

        return predictions_

    def _create_window_generator(self, embedding_info, output_config):

        window = EnhancedWindowGenerator(
            input_width=self.model_config['input_width'],
            label_width=self.model_config['output_width'],
            shift=self.model_config['shift'],
            label_columns=list(self.model_config['output_config'].keys()),
            numeric_columns=self.model_config['numeric_columns'],
            categorical_columns=self.model_config['categorical_columns'],
            embedding_configs=embedding_info,
            output_configs=output_config
        )

        return window

    def reconstruct_model(self):
        """重构用于预测的干净模型"""

        if not hasattr(self, 'best_checkpoint'):
            raise ValueError('未找到最佳模型检查点')  # 现在改为分片 / 训练里面也有

        checkpoint_dir = self.best_checkpoint
        keras_file = os.path.join(checkpoint_dir, 'model.keras')  # 现在保存的是.keras格式，需要找到具体的.keras文件

        if not os.path.exists(keras_file):
            raise FileNotFoundError(
                f"找不到.keras模型文件: {keras_file}\n"
                f"目录内容: {os.listdir(checkpoint_dir)}"
            )
        model = tf.keras.models.load_model(keras_file)

        # 重新编译（用于预测）
        self._compile_for_prediction_model(model)

        return model

    def evaluate_model(self, dataset, dataset_type='val'):
        """用任意数据评估已训练好的模型"""
        if not self._prediction_model:
            model = self.reconstruct_model()
        else:
            model = self._prediction_model

        metrics = ModelEvaluation(self.model_config['output_config'], model_name=self.model_config['model_type'])
        details = metrics.comprehensive_model_evaluation(model=model,  # 评估 best_model
                                                         window=self.window,
                                                         dataset=dataset,
                                                         dataset_type=dataset_type)
        return details

    def _generate_future_timestamps(self, last_time, n_steps, freq):
        return pd.date_range(start=last_time + self.model_config['shift'], periods=n_steps, freq=6 * freq)

    def _compile_for_prediction_model(self, model):  # 同一Python进程中直接获取实例。独立的演化路径
        """为预测模型重新编译 多输出会折叠metrics会折叠"""

        # 获取实际输出数量
        num_outputs = len(model.outputs)
        logger.debug(f"模型有 {num_outputs} 个输出")

        # 获取输出层名称（使用模型输出层名称，不是张量名称）
        output_names = []
        for output in model.outputs:
            for layer in model.layers:
                if hasattr(layer, 'output') and layer.output is output:
                    output_names.append(layer.name)
                    break
        logger.debug(f"输出层名称：{output_names}")

        # 构建字典配置
        # 使用统一的配置管理器
        loss_config = ModelConfigManager.get_loss_config(self.model_config)
        metrics_config = ModelConfigManager.get_metrics_config(self.model_config)
        loss_weights_config = ModelConfigManager.get_loss_weights_config(self.model_config)

        logger.debug(f"loss_config: {loss_config}")
        logger.debug(f"metrics_config: {metrics_config}")
        logger.debug(f"loss_weights_config:{loss_weights_config}")

        # 获取优化器
        if hasattr(self, 'training_model_') and hasattr(self.training_model_, 'optimizer'):
            optimizer = self.training_model_.optimizer  # 可以用实例，load可以用配置
        else:
            learning_rate = self.model_config.get('learning_rate', 0.001)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # 单输出或者多输出都可以使用字典，但是要保证输出层名字正确
        logger.debug("=== 编译前检查 ===")
        logger.debug(f"输出层: {output_names}")
        logger.debug(f"loss_config: {loss_config}")
        logger.debug(f"metrics_config: {metrics_config}")
        logger.debug(f"loss_config类型: {type(loss_config)}")
        logger.debug(f"metrics_config类型: {type(metrics_config)}")

        model.compile(
            optimizer=optimizer,
            loss=loss_config,  # 字典 键是输出层名
            loss_weights=loss_weights_config,
            metrics=metrics_config
        )

        logger.debug("编译完成，验证metrics配置...")

        if len(model.metrics) >= 2:
            compile_metrics = model.metrics[1]
            if hasattr(compile_metrics, '_user_metrics'):
                actual_metrics = compile_metrics._user_metrics
                logger.debug(f"实际编译的metrics配置: {actual_metrics}")
                logger.debug(f"期望的metrics配置: {metrics_config}")

        return model

    def _get_compile_config_for_save(self):

        # 1. 获取优化器实例
        if hasattr(self, 'training_model_') and self.training_model_.optimizer:
            optimizer_config = self.training_model_.optimizer.get_config()
            # 磁盘恢复传递字典get_config() 获取可序列化的配置 配置字典（但加载的时候需要实例）
            # 'optimizer': {'class_name': 'Adam', 'config': {...}},
        else:
            # 回退逻辑
            learning_rate = self.model_config.get('learning_rate', 0.001)
            default_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            optimizer_config = default_optimizer.get_config()

        # 2. 获取输出层名称
        loss_config = ModelConfigManager.get_loss_config(self.model_config)
        metrics_config = ModelConfigManager.get_metrics_config(self.model_config)
        loss_weights_config = ModelConfigManager.get_loss_weights_config(self.model_config)

        # 3. 构建字典格式的loss和metrics配置
        return {
            'optimizer': optimizer_config,
            'loss': loss_config,
            'metrics': metrics_config,
            'loss_weights': loss_weights_config,
            'output_names': list(self.model_config.get('output_config', {}).keys())  # 额外保存输出层名称，方便对齐
        }

    def clear_prediction_cache(self):
        """清空预测缓存"""
        if hasattr(self, '_prediction_model'):
            del self._prediction_model

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

    def __getstate__(self):
        """序列化时只保留必要信息"""
        state = self.__dict__.copy()

        # 移除所有模型实例（通过save/load机制重建）
        state['training_model_'] = None
        state['prediction_model_'] = None
        state['window'] = None

        return state

    def __setstate__(self, state):
        """反序列化"""
        self.__dict__.update(state)

        if hasattr(self, 'weights_path') and os.path.exists(self.weights_path):
            self.prediction_model_ = self.reconstruct_model()
            self.window = self._create_window_generator(self.embedding_info, self.model_config['output_config'])

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


class EmbeddingConfig:
    """Embedding维度选择配置"""

    @staticmethod
    def get_embedding_dim(n_categories: int) -> int:
        if n_categories <= 2:
            return 1  # 二分类
        elif n_categories <= 5:
            return max(1, min(2, n_categories - 1))  # 小类别
        elif n_categories <= 10:
            return 3  # 中等类别
        elif n_categories <= 20:
            return 4  # 较大类别
        elif n_categories <= 50:
            return 6  # 大类别
        else:
            # 谷歌研究公式：1.6 * n_categories^0.56
            return min(50, int(1.6 * n_categories ** 0.56))  # 保守的公式 min(8, int(0.8 * n_categories ** 0.4))

    @staticmethod
    def should_use_embedding(n_categories: int, unique_ratio: float) -> bool:
        """
        判断是否应该使用Embedding
        unique_ratio: 唯一值数量 / 总样本数
        """

        if n_categories <= 5:
            return unique_ratio < 0.1
        elif n_categories <= 20:  # 高基数或中等基数都推荐使用Embedding
            return unique_ratio < 0.3
        else:
            return unique_ratio < 0.5

    @staticmethod
    def _get_embedding_info(dataset: pd.DataFrame, cat_cols: list):
        embedding_configs: Dict[str, Dict[str, Any]] = {}

        if cat_cols and isinstance(dataset, pd.DataFrame):

            for col in cat_cols:
                series = dataset[col].dropna()
                n_categories = int(series.nunique())
                unique_ratio = n_categories / len(series)

                base_config: Dict[str, Any] = {
                    'input_dim': n_categories + 1,  # 已经预留了一个__UNKNOWN__ 不代表训练集就有n+1个种类，算的是(实际数据集中的种类数+1)
                    'name': f'embedding_{col}'
                }

                if EmbeddingConfig.should_use_embedding(n_categories, unique_ratio):
                    base_config['output_dim'] = EmbeddingConfig.get_embedding_dim(n_categories)

                    # 自动添加正则化
                    if n_categories <= 10:  # 小到中等类别
                        base_config['embeddings_regularizer'] = l2(0.001)

                else:  # 轻量Embedding
                    base_config.update({
                        'output_dim': max(1, min(2, n_categories // 20)),
                        'embeddings_regularizer': l2(0.1)  # 强正则化(output_dim 从3->1)
                    })
                embedding_configs[col] = base_config

            return embedding_configs


class ModelConfigManager:
    """统一管理模型配置的辅助类"""

    @staticmethod
    def get_loss_config(model_config):
        # 单和多输出，都是字典
        output_config = model_config.get('output_config', {})

        output_names = list(output_config.keys())
        loss_config = {}
        for output_name in output_names:
            cfg = output_config.get(output_name, {})
            loss_type = cfg.get('type', 'regression')
            loss_config[output_name] = cfg.get('loss', ModelConfigManager._get_default_loss(loss_type))
        return loss_config

    @staticmethod
    def get_metrics_config(model_config):
        output_config = model_config.get('output_config', {})

        output_names = list(output_config.keys())
        metrics_config = {}
        for output_name in output_names:
            cfg = output_config.get(output_name, {})
            loss_type = cfg.get('type', 'regression')
            metrics = cfg.get('metrics', ModelConfigManager._get_default_metrics(loss_type))
            metrics_config[output_name] = metrics if isinstance(metrics, list) else [metrics]

        return metrics_config

    @staticmethod
    def get_loss_weights_config(model_config):
        output_config = model_config.get('output_config', {})

        output_names = list(output_config.keys())
        loss_weights = {}
        for output_name in output_names:
            cfg = output_config.get(output_name, {})
            loss_weights[output_name] = cfg.get('loss_weights', 1.0)
        return loss_weights

    @staticmethod
    def _get_default_loss(loss_type):
        defaults = {
            'regression': 'mse',
            'classification': 'sparse_categorical_crossentropy',
            'binary_classification': 'binary_crossentropy'
        }
        return defaults.get(loss_type, 'mse')

    @staticmethod
    def _get_default_metrics(loss_type):
        defaults = {
            'regression': ['mae'],
            'classification': ['accuracy'],
            'binary_classification': ['accuracy']
        }
        return defaults.get(loss_type, ['mae'])
