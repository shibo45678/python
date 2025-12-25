import json
import pickle
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Any
import re
import joblib
import shutil

import numpy as np
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
"""    一次训练直接 predict（keras格式），
       隔日预测 TrainedModelPredictor(new_data)（keras格式） ，
       专用部署 (格式Saved Model) """


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
        self._prediction_model = self.load_best_model()

        # 1.5 评估模型
        self.evaluate_model(dataset=val_window_data, dataset_type='val')

        self.is_fitted_ = True

        return self

    @validate_input(validate_y=False)
    def predict(self, X):
        check_is_fitted(self)

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_model = X_[self.input_cols_]

        # 1. 处理窗口数据
        predict_window_data = self.window.createMultimodalDataset(X_model)

        # 2. 重构模型
        if self.prediction_model_ is None:
            self._prediction_model = self.load_best_model()  # 确保使用最佳权重

        # 3. 模型预测
        predictions = self._prediction_model.predict(predict_window_data)  # 多输入和输出（tuple,dict）->预测结果是list

        return predictions

    def load_best_model(self):
        """重构用于预测的干净模型"""

        if not hasattr(self, 'best_checkpoint'):
            raise ValueError('未找到最佳模型检查点')

        checkpoint_dir = self.best_checkpoint
        keras_file = os.path.join(checkpoint_dir, 'model.keras')  # 现在保存的是.keras格式，需要找到具体的.keras文件

        if not os.path.exists(keras_file):
            raise FileNotFoundError(
                f"找不到.keras模型文件: {keras_file}\n"
                f"目录内容: {os.listdir(checkpoint_dir)}"
            )
        model = tf.keras.models.load_model(keras_file)  # 可以直接 predict() evaluate() 甚至可以继续训练（如果有优化器状态）

        logger.debug("\n加载的模型:")
        logger.debug(f"  优化器: {model.optimizer}")
        logger.debug(f"  Loss: {model.loss}")
        logger.debug(f"  Metrics: {model.metrics}")  # 多任务的metrics 也可以打开 <CompileMetrics name=compile_metrics>]

        # 重新编译 展开多任务metrics 评估<CompileMetrics name=compile_metrics>]
        # self._compile_for_prediction_model(model)

        return model

    def evaluate_model(self, dataset, dataset_type='val'):
        """用任意数据评估已训练好的模型"""
        if not self._prediction_model:
            model = self.load_best_model()
        else:
            model = self._prediction_model

        metrics = ModelEvaluation(self.model_config['output_config'], model_name=self.model_config['model_type'])
        details = metrics.comprehensive_model_evaluation(model=model,  # 评估 best_model
                                                         window=self.window,
                                                         dataset=dataset,
                                                         dataset_type=dataset_type)
        return details

    def clear_prediction_cache(self):
        """清空预测缓存"""
        if hasattr(self, '_prediction_model'):
            del self._prediction_model

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
            self.prediction_model_ = self.load_best_model()
            self.window = self._create_window_generator(self.embedding_info, self.model_config['output_config'])


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


class TimeSeriesPostProcessor:
    """
    功能：
    1. 时间戳生成和拼接
    2. 逆转换（标准化还原）
    3. 多任务结果处理
    4. 状态保存和加载(预处理的cleaner/pipeline
    args: config 包括：
                 'model_name':model_name,
                 'preprocessor':preprocessor,
                 'save_dir': save_dir,
                 'task_names':config.get('output_config').keys().tolist(),
                 'output_width':config.get('output_width',1),
                 'time_col_name'：引用原列名
    """

    @validate_arguments
    def __init__(self, config: Dict = Field(..., description="配置字典，包含freq、shift等信息")):
        self.config = config
        self.serialized_states = {}  # pipeline序列化状态
        self._temp_preprocessor = None  # 受保护（约定上外部不应直接访问

    def capture_and_save_pipeline_state(self):
        """
        捕获并立即序列化保存pipeline状态
        Args:
            preprocessor: 预处理器对象
            save_dir: 保存目录（如果为None，仅保存在内存中）...
        """
        # 1. 保存临时引用
        self._temp_preprocessor = self.config.get('preprocessor', None)

        # 2. 提取并序列化状态
        serialized_states = {}
        if hasattr(self._temp_preprocessor, 'pipelines_'):
            for pipe_name, pipeline in self._temp_preprocessor.pipelines_.items():
                serialized_states[pipe_name] = {}

                for step_name, transformer in pipeline.named_steps.items():
                    try:
                        serialized = pickle.dumps(transformer)
                        serialized_states[pipe_name][step_name] = {
                            'pickled': serialized,
                            'type': type(transformer).__name__,
                            'params': transformer.get_params() if hasattr(transformer, 'get_params') else {}
                        }
                    except Exception as e:
                        logger.info(f"警告：无法pickle{pipe_name}.{step_name}:{e}")
                        # 如果pickle失败，只保存关键属性
                        serialized_states[pipe_name][step_name] = {
                            'pickled': None,
                            'type': type(transformer).__name__,
                            'attributes': self._extract_critical_attributes(transformer)
                        }
            self.serialized_states = serialized_states

            # 3. 如果指定了保存目录，立即写入磁盘
            save_dir = self.config.get('save_dir', '/Users/shibo/Python/NeuralNetwork/saved_model_state')
            if save_dir:
                self._save_to_disk(save_dir)

            return self

    def _extract_critical_attributes(self, transformer):
        attrs = {}

        if hasattr(transformer, 'scaling_config_'):
            attrs['scaling_config_'] = transformer.scaling_config_.tolist() if hasattr(transformer.scaling_config_,
                                                                                       'tolist') else transformer.scaling_config_
        if hasattr(transformer, 'encoders_'):
            attrs['encoders_'] = transformer.encoders_

        return attrs

    def _save_to_disk(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_file = Path(save_dir) / 'pipeline_states.pkl'

        save_data = {
            'serialized_states': self.serialized_states,
            'config': self.config,
            'saved_at': datetime.now().isoformat()
        }
        with state_file.open('wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Pipeline状态已保存到: {state_file}")
        return save_data

    def custom_inverse_transform(self, raw_predictions, use_saved, target_columns, **kwargs):
        """
        智能逆转换：根据情况选择使用内存引用或保存的状态
        支持多预测的逆转换

        Args:
           raw_predictions: 原始预测
           use_saved: True=使用保存的状态，False=尝试使用内存引用
           **kwargs: 其他参数

        Returns:
           逆转换后的结果
        """
        if isinstance(raw_predictions, (list, tuple)):
            logger.debug(f"[INFO] 多任务预测: {len(raw_predictions)}个任务")

            processed_tasks = []
            for i, task_pred in enumerate(raw_predictions):
                logger.debug(f"[INFO] 任务{i}原始形状: {task_pred.shape}")

                # 去除冗余的最后一个维度(samples, output_width,1)
                if task_pred.shape[-1] == 1:
                    task_pred = task_pred.squeeze(-1)
                    logger.debug(f"[INFO] 任务{i}去除冗余后: {task_pred.shape}")
                    column_names = [f'pred_{target_columns[i]}_{j}' for j in range(task_pred.shape[1])]
                    task_pred = pd.DataFrame(task_pred,
                                             columns=column_names)  # n = samples, columns = [pred_T_0,pred_T_1...]

                if not use_saved and hasattr(self, '_temp_preprocessor'):
                    task_result = self._inverse_transform_live(predictions=task_pred, target_column=target_columns[i],
                                                               **kwargs)
                else:
                    task_result = self._inverse_transform_from_saved(predictions=task_pred,
                                                                     target_column=target_columns[i], **kwargs)

                logger.debug(f"[INFO] 任务{i}逆转换后: {task_result.shape}")
                processed_tasks.append(task_result)
            return processed_tasks

        else:
            if not use_saved and hasattr(self, '_temp_preprocessor'):
                return self._inverse_transform_live(raw_predictions, **kwargs)
            else:
                return self._inverse_transform_from_saved(raw_predictions, **kwargs)

    def _inverse_transform_live(self, predictions: pd.DataFrame, pipeline_name='pipeline_4',
                                step_names=None, target_column: str = None):

        logger.debug(f"[DEBUG] _inverse_transform_live 开始")
        logger.debug(f"输入形状: {predictions.shape}")
        logger.debug(f"pipeline_name: {pipeline_name}")
        logger.debug(f"step_names: {step_names}")
        logger.debug(f"target_column: {target_column}")

        if step_names is None:
            step_names = ['engineer_3', 'engineer_4']

        result = predictions

        for step_name in step_names:
            transformer = self._temp_preprocessor.pipelines_[pipeline_name].named_steps[step_name]

            if step_name == 'engineer_3':
                valid_col = transformer.numeric_columns_
                if target_column is not None and target_column in valid_col:  # 只有数值列才进行标准化
                    result = transformer.custom_inverse_transform(scaled_data=result,
                                                                  target_column=target_column)  # 更新result
                else:
                    logger.debug(f"目标列{target_column}不需要数值的逆标准化转换")

            elif step_name == 'engineer_4':
                valid_col = transformer.categorical_columns_
                if target_column is not None and target_column in valid_col:  # 只有分类列才进行编码
                    result = transformer.custom_inverse_transform(scaled_data=result, target_column=target_column)
                else:
                    logger.debug(f"目标列{target_column}不需要分类列的逆编码转换")

            return result

    def _inverse_transform_from_saved(self, predictions, pipeline_name='pipeline_4',
                                      step_names=None, target_column=None):

        if step_names is None:
            step_names = ['engineer_3', 'engineer_4']

        result = predictions

        for step_name in step_names:
            if pipeline_name in self.serialized_states and step_name in self.serialized_states[pipeline_name]:
                state_info = self.serialized_states[pipeline_name][step_name]

                # 从pickle重建transformer
                if state_info.get('pickled'):
                    transformer = pickle.loads(state_info['pickled'])

                    if hasattr(transformer, 'inverse_transform'):
                        if step_name == 'engineer_3':
                            valid_col = transformer.numeric_columns_
                            if target_column is not None and target_column in valid_col:
                                result = transformer.custom_inverse_transform(scaled_data=result,
                                                                              target_column=target_column)
                            else:
                                logger.debug(f"目标列{target_column}不需要数值的逆标准化转换")

                        else:
                            valid_col = transformer.categorical_columns_
                            if target_column is not None and target_column in valid_col:
                                result = transformer.custom_inverse_transform(scaled_data=result,
                                                                              target_column=target_column)
                            else:
                                logger.debug(f"目标列{target_column}不需要分类列的逆编码转换")
                else:
                    logger.debug(f"pickled失败需要手动")

        logger.info(f"最终的数据：{result.tail(10)}")
        return result

    def add_timestamps(self, predictions, historical_timestamps, input_width: int, output_width:int,shift: int, freq: str):
        """
        参数:
            historical_timestamps: 预测数据的 历史时间戳  datetime64 处理后的 （长度7009）
            input_width: 输出数据时间步
            shift：偏移的时间步 24
            freq: 时间频率（'h'）
        返回:
            windows_start_times: 每个窗口的基准时间
            forecast_timestamps: 每个窗口的预测时间点
        """

        # 预测样本数（只需要输入）
        # n_windows = len(historical_timestamps) - input_width - shift + 1

        # 训练窗口数（有真实标签的） len -total_window + 1 =1548 - 34 +1 =1515
        n_windows = len(historical_timestamps) - (input_width + shift  + output_width- 1) +1

        window_start_times = []  # 每个窗口的基准时间
        future_timestamps = []  # 每个窗口的预测时间点列表

        # 先检查数据
        # print("=== 数据检查 ===")
        # print(f"数据总长度: {len(historical_timestamps)}")
        # print(f"数据时间范围: {historical_timestamps.iloc[0]} 到 {historical_timestamps.iloc[-1]}")

        # 计算窗口
        # n_windows = len(historical_timestamps) - (input_width + shift  + output_width- 1) +1
        # print(f"\n=== 窗口计算 ===")
        # print(f"input_width: {input_width}")
        # print(f"计算出的窗口数: {n_windows}")
        #
        # # 检查第一个和最后一个窗口
        # print(f"\n第一个窗口:")
        # first_start_idx = 0
        # first_end_idx = first_start_idx + input_width - 1
        # print(f"  索引: {first_start_idx} 到 {first_end_idx}")
        # print(f"  时间: {historical_timestamps.iloc[first_start_idx]} 到 {historical_timestamps.iloc[first_end_idx]}")
        #
        # print(f"\n最后一个窗口:")
        # last_start_idx = n_windows - 1


        for i in range(n_windows):  # 预测不需要有真实值的窗口 最后1个i位置：len-input+1-1
            last_time = historical_timestamps.iloc[i + input_width - 1]  # 输入窗口的最后一条
            window_start_times.append(last_time)


            # 从base_time + shift(时间步）开始预测
            future_time = self._generate_future_timestamps(last_time,
                                                           n_steps=self.config.get('output_width', 1),
                                                           freq=freq,
                                                           shift=shift)
            future_timestamps.append(future_time)
            #  Timestamp('2016-10-28 18:00:00') /
            #  DatetimeIndex(['2016-10-29 18:00:00', '2016-10-29 19:00:00', '2016-10-29 20:00:00', '2016-10-29 21:00:00', '2016-10-29 22:00:00'], dtype='datetime64[ns]', freq='h')

        print("window_start_times 验证:")
        print(f"长度: {len(window_start_times)}")
        print(f"第一个: {window_start_times[0]}")
        print(f"最后一个: {window_start_times[-1]}")
        print(f"应该是: {historical_timestamps.iloc[-5]} - 24小时")

        return self._create_result_df(predictions, window_start_times, future_timestamps)

    def _generate_future_timestamps(self, last_time, n_steps: int, freq: str, shift: int):
        # 将 shift=24 时间步* 毎步间隔1小时= 转换为时间增量24小时，并确保单位与 freq 1h 的小时匹配

        if isinstance(shift, (int, float)):
            match = re.match(r'(\d+)', freq)
            if match:
                freq_num = int(match.group(1))
            else:
                freq_num = 1

            if 'h' in freq.lower():
                time_shift = pd.Timedelta(hours=shift * freq_num)
            elif 'D' in freq:
                time_shift = pd.Timedelta(days=shift * freq_num)
            elif 'min' in freq:
                time_shift = pd.Timedelta(minutes=shift * freq_num)
            else:
                # 默认使用 freq 的单位，但需要解析 freq 字符串
                time_shift = shift * pd.Timedelta(freq)

        elif isinstance(shift, pd.Timedelta):
            time_shift = shift
        else:
            time_shift = pd.Timedelta(0)

        start = last_time + time_shift
        return pd.date_range(start=start, periods=n_steps, freq=freq)

    def _create_result_df(self, predictions, window_start_times: list, future_timestamps: list):
        """单任务和多任务区分（单：1个数组，多：每个元素是一个任务的输出
            df三列：开始时间、预测时间列、任务1，任务2"""
        task_names = self.config.get('task_names')

        predictions_dict = {}
        if isinstance(predictions, list):  # 多任务
            for i, pred in enumerate(predictions):
                task_name = task_names[i] if i < len(task_names) else f'task_{i}'
                predictions_dict[task_name] = pred

        elif predictions.ndim == 3:  # 单输出但多步：最后一个维度是任务维度(多分类) 待确认
            num_tasks = predictions.shape[2]
            for i in range(num_tasks):
                task_name = task_names[i] if i < len(task_names) else f'task_{i}'
                predictions_dict[task_name] = predictions[:, :, i]
        else:
            # 其他格式：
            if len(task_names) > 0:
                predictions_dict[task_names[0]] = predictions
            else:
                predictions_dict['prediciton'] = predictions

        num_windows = len(predictions[0])
        print(num_windows)

        all_windows = []
        for i in range(num_windows):  # 窗口数量
            start_times = window_start_times[i]
            future_times = future_timestamps[i]

            for step in range(self.config.get('output_width', 1)):
                window = {
                    'window_end': start_times,
                    'forecast_time': future_times[step]}
                window.update(
                    **{f'{task_name}_pred': pred_values.iloc[i].iloc[step] for task_name, pred_values in
                       predictions_dict.items()}  # 窗口定位 i
                )

                all_windows.append(window)

        print(all_windows[-1])  # {'window_end': Timestamp('2016-12-31 00:00:00'), 'forecast_time': Timestamp('2017-01-01 04:00:00'), 'T_pred': -1.5133829, 'rh_pred': 87.85305}

        results_df = pd.DataFrame(all_windows)  # pd.concat 是组合df的，但这里是字典

        logger.debug(f"生成的预测记录总数: {len(results_df)}")  # 6980×5=34900
        logger.debug(f"CSV文件预览:")
        logger.debug(results_df.head(10))

        results_df.to_csv(
            '/Users/shibo/Python/NeuralNetwork/temperature_forecasting/data/intermediate/predictions_result.csv',
            index=False)
        return results_df
        # T_actual  merge / 少最后一个数据点

        # # else:
        # # 单任务
        # flat_pred = self._flatten_prediction(predictions, len(timestamps))
        # df = pd.DataFrame({
        #     self.config.get('time_col_name', 'Time'): timestamps,
        #     'prediction': flat_pred
        # })
        #
        # all_rows = []
        #
        # for i in range(len(predictions[0])):  # 窗口数量
        #

    #
    #         return df
    #
    # # df = pd.DataFrame({
    # #     self.config.get('time_col_name', 'Date Time'): timestamps
    # # })
    # # steps_ahead = len(timestamps)
    # #
    #
    # # def _flatten_prediction(self, prediction, n_steps):
    # #     if hasattr(prediction, 'flatten'):
    # #         flat = prediction.flatten()
    # #     else:
    # #         flat = np.array(prediction).flatten()
    # #     # 确保长度匹配
    # #     if len(flat) >= n_steps:
    # #         return flat[:n_steps]
    # #     elif len(flat) < n_steps:
    # #         return np.pad(flat, (0, n_steps - len(flat)),
    # #                       mode='constant', constant_values=np.nan)
    # #     return flat
    #
    # # def save_state(self, save_dir):
    # #     """保存后处理器状态（用于场景2、3）"""
    # #     os.makedirs(save_dir, exist_ok=True)
    # #
    # #     # 1. 保存状态
    # #     state_file = os.path.join(save_dir, 'postprocessor_state.pkl')
    # #     with open(state_file, 'wb') as f:
    # #         pickle.dump({
    # #             'config': self.config,
    # #             'pipeline_states': self.serialized_states,
    # #             'scaler_states': self.scaler_states,
    # #             'saved_at': datetime.now().isoformat()
    # #         }, f)
    # #
    # #     # 2. 保存配置
    # #     config_file = os.path.join(save_dir, 'postprocessor_config.json')
    # #     with open(config_file, 'w') as f:
    # #         json.dump(self.config, f, indent=2)
    # #
    # #     logger.info(f"后处理器状态已保存到: {save_dir}")
    # #
    # # def calculate_val_mape(self):
    # #     pass
    #


# def _extract_transformer_state(self, transformer):

# @classmethod
# def load_state(cls, save_dir):
#     """加载后处理器状态（用于场景2、3）"""
#     state_file = os.path.join(save_dir, 'postprocessor_state.pkl')
#
#     if not os.path.exists(state_file):
#         raise FileNotFoundError(f"状态文件不存在: {state_file}")
#
#     with open(state_file, 'rb') as f:
#         state_data = pickle.load(f)
#
#     # 创建实例
#     processor = cls(state_data['config'])
#     processor.pipeline_states = state_data['pipeline_states']
#     processor.scaler_states = state_data.get('scaler_states', {})
#
#     print(f"后处理器状态已从 {save_dir} 加载")
#     return processor
#
# def create_deployment_package(self, model, save_dir='deployment_package'):
#     """创建完整的部署包（用于场景3）"""
#     import joblib
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 1. 保存模型
#     if hasattr(model, 'save'):
#         # Keras模型
#         model.save(os.path.join(save_dir, 'model.keras'))
#     else:
#         # 其他类型模型
#         joblib.dump(model, os.path.join(save_dir, 'model.joblib'))
#
#     # 2. 保存后处理器状态
#     self.save_state(save_dir)
#
#     # 3. 保存部署配置
#     deployment_config = {
#         'model_type': type(model).__name__,
#         'input_shape': getattr(model, 'input_shape', None),
#         'output_shape': getattr(model, 'output_shape', None),
#         'postprocessor_config': self.config,
#         'created_at': datetime.now().isoformat(),
#         'usage_example': self._create_usage_example()
#     }
#
#     config_file = os.path.join(save_dir, 'deployment_config.json')
#     with open(config_file, 'w') as f:
#         json.dump(deployment_config, f, indent=2)
#
#     print(f"部署包已创建: {save_dir}")
#
# def _create_usage_example(self):
#     """创建使用示例代码"""
#     return '''
# # 使用部署包进行预测
# from timeseries_postprocessor import TimeSeriesPostProcessor
# import tensorflow as tf
# import pandas as pd
#
# # 1. 加载模型
# model = tf.keras.models.load_model('model.keras')
#
# # 2. 加载后处理器
# postprocessor = TimeSeriesPostProcessor.load_state('.')
#
# # 3. 预测
# raw_predictions = model.predict(new_data)
#
# # 4. 逆转换
# inverse_predictions = postprocessor.inverse_transform(
#    raw_predictions,
#             use_saved=False,  # 使用内存中的preprocessor
#             pipeline_name='pipeline_4',
#             step_names=['engineer_3', 'engineer_4'],
#             target_columns=['T', 'rh']
# )
#
def save_state(self, save_dir):
    """保存后处理器状态（用于场景2、3）"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存状态
    state_file = os.path.join(save_dir, 'postprocessor_state.pkl')
    with open(state_file, 'wb') as f:
        pickle.dump({
            'config': self.config,
            'pipeline_states': self.pipeline_states,
            'scaler_states': self.scaler_states,
            'saved_at': datetime.now().isoformat()
        }, f)

    # 2. 保存配置
    config_file = os.path.join(save_dir, 'postprocessor_config.json')
    with open(config_file, 'w') as f:
        json.dump(self.config, f, indent=2)

    print(f"后处理器状态已保存到: {save_dir}")


@classmethod
def load_state(cls, save_dir):
    """加载后处理器状态（用于场景2、3）"""
    state_file = os.path.join(save_dir, 'postprocessor_state.pkl')

    if not os.path.exists(state_file):
        raise FileNotFoundError(f"状态文件不存在: {state_file}")

    with open(state_file, 'rb') as f:
        state_data = pickle.load(f)

    # 创建实例
    processor = cls(state_data['config'])
    processor.pipeline_states = state_data['pipeline_states']
    processor.scaler_states = state_data.get('scaler_states', {})

    print(f"后处理器状态已从 {save_dir} 加载")
    return processor


def create_deployment_package(self, model, save_dir='deployment_package'):
    """创建完整的部署包（用于场景3）"""
    import joblib

    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存模型
    if hasattr(model, 'save'):
        # Keras模型
        model.save(os.path.join(save_dir, 'model.keras'))
    else:
        # 其他类型模型
        joblib.dump(model, os.path.join(save_dir, 'model.joblib'))

    # 2. 保存后处理器状态
    self.save_state(save_dir)

    # 3. 保存部署配置
    deployment_config = {
        'model_type': type(model).__name__,
        'input_shape': getattr(model, 'input_shape', None),
        'output_shape': getattr(model, 'output_shape', None),
        'postprocessor_config': self.config,
        'created_at': datetime.now().isoformat(),
        'usage_example': self._create_usage_example()
    }

    config_file = os.path.join(save_dir, 'deployment_config.json')
    with open(config_file, 'w') as f:
        json.dump(deployment_config, f, indent=2)

    print(f"部署包已创建: {save_dir}")


def _create_usage_example(self):
    """创建使用示例代码"""
    return '''
# 使用部署包进行预测
from timeseries_postprocessor import TimeSeriesPostProcessor
import tensorflow as tf
import pandas as pd

# 1. 加载模型
model = tf.keras.models.load_model('model.keras')

# 2. 加载后处理器
postprocessor = TimeSeriesPostProcessor.load_state('.')

# 3. 预测
raw_predictions = model.predict(new_data)

# 4. 逆转换
inverse_predictions = postprocessor.inverse_transform(
    raw_predictions,
    pipeline_name='pipeline_4',
    step_names=['engineer_3', 'engineer_4']
)

# 5. 添加时间戳
results = postprocessor.add_timestamps(
    inverse_predictions,
    historical_timestamps=historical_timestamps,
    freq='6H'
)
'''

    # def _get_last_timestamp(self, timestamps):
    #     """获取最后一个时间戳"""
    #     if hasattr(timestamps, 'iloc'):
    #         last = timestamps.iloc[-1]
    #         logger.debug(f"最后一个时间戳：{last}")
    #         return last
    #     elif isinstance(timestamps, (list, np.ndarray)):
    #         last = timestamps[-1]
    #         logger.debug(f"最后一个时间戳：{last}")
    #         return last
    #     else:
    #         last = timestamps
    #         logger.debug(f"最后一个时间戳：{last}")
    #         return last
