import json
import random

import cloudpickle
import pickle
import warnings
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Any, List, BinaryIO
import re
import datetime
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
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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
        self.train_window_data_ = None
        self.val_window_data_ = None
        self.test_window_data_ = None
        self.input_cols_ = None  # 处理掉时间列，保证进入模型的所有列是数值
        self.forecast_window_gen_ = None
        self.train_window_gen_ = None
        self.forecast_window_config_ = None
        self.stage_number_=-1

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
        self.embedding_info_ = EmbeddingConfig._get_embedding_info(train_datasets_,  # 原始DF
                                                                   self.model_config['categorical_columns']
                                                                   )

        # 1.2 处理窗口数据
        self.train_window_gen_ = self._train_window_generator(self.model_config['output_config'])
        train_window_data = self.train_window_gen_.createDataset(train_datasets_)
        val_window_data = self.train_window_gen_.createDataset(val_datasets_)

        continue_train = self.model_config.get('continue_from', None)

        # 首次训练
        if continue_train is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            basic_dir = f"saved_model/{self.model_config['model_type']}_{timestamp}"
            os.makedirs(basic_dir, exist_ok=True)  # 存模型最佳检查点

            continue_training_dir = os.path.join(basic_dir, 'continue_training')  # 继续训练文件夹
            os.makedirs(continue_training_dir, exist_ok=True)

            self._save_preprocessed_data(continue_training_dir=continue_training_dir, trainset=train_window_data,
                                         valset=val_window_data)
            self.stage_number_ = -1

        # ====继续训练用====
        else:
            basic_dir = None
            match = re.search(r"(.*)tf_checkpoints_stage(\d+)", str(continue_train))
            continue_training_dir = os.path.join(match.group(1),'continue_training')
            self.stage_number_ = int(match.group(2))

        # ================

        # 1.3 多任务/单任务分支
        if self.model_config['multi_tasks']:

            if self.model_config['model_type'].startswith('multi_lstm'):
                model_config = {**self.model_config,
                                'embedding_configs': self.embedding_info_}

                lstm_model = MultiTasksLstmModel(model_config)
                lstm_model_ = lstm_model._build_lstm_model()
                self.training_model_ = lstm_model_

            elif self.model_config['model_type'].startswith('multi_cnn'):
                model_config = {**self.model_config,  # 解包
                                'embedding_configs': self.embedding_info_}  # 追加
                cnn_model = MultiTasksCnnModel(architecture_type='enhance_parallel', config=model_config)
                cnn_model_ = cnn_model._build_cnn_model()
                self.training_model_ = cnn_model_

            else:
                raise ValueError(f"未支持的模型{self.model_config['model_type']}")
            # 1.4 训练模型
            self.history_, best_checkpoint = TrainingMultiModel(model_name=self.model_config['model_type'],
                                                                model=self.training_model_, trainset=train_window_data,
                                                                valset=val_window_data,
                                                                basic_dir=basic_dir,
                                                                total_epochs=self.model_config['total_epochs'],
                                                                verbose=self.model_config['verbose'],
                                                                monitor=self.model_config['monitor'],
                                                                min_delta=self.model_config[
                                                                    'min_delta'],
                                                                continue_from_experiment=self.model_config[
                                                                    'continue_from'])
        # 单任务
        else:
            if self.model_config['model_type'].startswith('single_lstm'):
                model_config = {**self.model_config,
                                'embedding_configs': self.embedding_info_}
                lstm_model = SingleTaskLstmModel(model_config)
                lstm_model_ = lstm_model._build_lstm_model()
                self.training_model_ = lstm_model_
            else:
                raise ValueError(f"未支持的模型{self.model_config['model_type']}")

            self.history_, best_checkpoint = TrainingSingleModel(model_name=self.model_config['model_type'],
                                                                 model=self.training_model_, trainset=train_window_data,
                                                                 valset=val_window_data,
                                                                 basic_dir=basic_dir,
                                                                 total_epochs=self.model_config['total_epochs'],
                                                                 verbose=self.model_config['verbose'],
                                                                 early_stop_patience=self.model_config[
                                                                     'early_stop_patience'],
                                                                 min_delta=self.model_config[
                                                                     'min_delta'],
                                                                 monitor = self.model_config['monitor'],
                                                                 continue_from_experiment=self.model_config[
                                                                     'continue_from'])

        #  ====继续训练用====  存预处理数据 + 训练历史
        self._save_model_config(continue_training_dir=continue_training_dir, config=model_config,
                                stage_number=self.stage_number_)

        self._save_training_history(history=self.history_, continue_training_dir=continue_training_dir,
                                    stage_number=self.stage_number_)
        # ==================

        # 保存最佳检查点路径供后续使用
        self.best_checkpoint = best_checkpoint  # epoch/

        # 训练完成后，创建用于预测的模型
        self.prediction_model_ = self.load_best_model()

        # 1.5 评估模型
        self.evaluate_model(dataset=val_window_data, dataset_type='val')

        self.is_fitted_ = True

        return self

    @validate_input(validate_y=False)
    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_model = X_[self.input_cols_]

        # 1. 处理窗口数据
        predict_window_data = self.train_window_gen_.createDataset(X_model)

        # 2. 重构模型
        if self.prediction_model_ is None:
            self.prediction_model_ = self.load_best_model()  # 确保使用最佳权重

        # 3. 模型预测
        predictions = self.prediction_model_.predict(predict_window_data)  # 多输入和输出（tuple,dict）->预测结果是list

        return predictions

    def _save_preprocessed_data(self, continue_training_dir, trainset, valset):
        model_name = self.model_config['model_type']
        saved_data_path = os.path.join(continue_training_dir, f'{model_name}_preprocessed_data')
        os.makedirs(saved_data_path, exist_ok=True)

        train_save_path = os.path.join(saved_data_path, 'train_dataset')
        trainset.save(train_save_path)

        val_save_path = os.path.join(saved_data_path, 'val_dataset')
        valset.save(val_save_path)

    def _save_model_config(self, continue_training_dir, config, stage_number):
        model_name = config.get('model_type')
        saved_config_path = os.path.join(continue_training_dir, f'{model_name}_config_stage{stage_number+1}.cpkl')
        with open(saved_config_path, 'wb') as f:
            cloudpickle.dump(config, f)

    def _save_training_history(self, history, continue_training_dir, stage_number):
        model_name = self.model_config.get('model_type', 'unknown')
        history_path = os.path.join(continue_training_dir, f'{model_name}_history_stage{stage_number+1}.cpkl')
        csv_path = os.path.join(continue_training_dir, f'{model_name}_history_stage{stage_number+1}.csv')
        # history 是一个 Keras History 对象 不可以直接dump
        # history.history：字典 / history.params：字典（可序列化） / history.epoch：列表（可序列化） 其他不可序列化

        if hasattr(history, 'history'):
            history_dict = history.history if hasattr(history, 'history') else history
            epochs = history.epoch if hasattr(history, 'epoch') else list(range(len(history_dict.get('loss', []))))
            params = history.params if hasattr(history, 'params') else {}
        else:
            history_dict = history
            epochs = list(range(1, len(next(iter(history_dict.values()))) ))
            params = {}

        history_data = {
            'model_name': model_name,
            'history': history_dict,
            'epochs': [int(e) for e in epochs],
            'params': params,
            'stage': stage_number+1,
            'save_time': datetime.datetime.now().isoformat()
        }
        with open(history_path, 'wb') as f:
            cloudpickle.dump(history_data, f)

        df = pd.DataFrame(history_dict)
        df.insert(0, 'epoch', epochs)
        df.to_csv(csv_path, index=False)
        logger.info(f"训练历史保存到: {csv_path}")
        return history_path,csv_path

    def load_best_model(self):
        """用于预测的干净模型"""

        if not hasattr(self, 'best_checkpoint'):
            raise ValueError('未找到最佳模型检查点')

        checkpoint_dir = self.best_checkpoint
        file_list = os.listdir(checkpoint_dir)
        keras_files =[]

        for file in file_list:
            if file.endswith('.keras'):
                keras_file = os.path.join(checkpoint_dir,file)
                keras_files.append(keras_file)

        if keras_files:
            keras_files.sort(key=os.path.getmtime, reverse=True)
            model = tf.keras.models.load_model(keras_files[0])
            # logger.debug("\n加载的模型:"
            # logger.debug(f"  优化器: {model.optimizer}")
            # logger.debug(f"  Loss: {model.loss}")
            # logger.debug(f"  Metrics: {model.metrics}")  # 多任务的metrics 也可以打开 <CompileMetrics name=compile_metrics>]
            return model
        else:
            raise FileNotFoundError(
                f"找不到.keras模型文件: {keras_files}\n"
            )

    def evaluate_model(self, dataset, dataset_type='val'):
        """用任意数据评估已训练好的模型"""
        if not self.prediction_model_:
            model = self.load_best_model()
        else:
            model = self.prediction_model_

        metrics = ModelEvaluation(self.model_config['output_config'], model_name=self.model_config['model_type'])
        details = metrics.comprehensive_model_evaluation(model=model,  # 评估 best_model
                                                         window=self.train_window_gen_,
                                                         dataset=dataset,
                                                         dataset_type=dataset_type)
        return details

    def clear_prediction_cache(self):
        """清空预测缓存"""
        if hasattr(self, 'prediction_model_'):
            del self.prediction_model_

    def _train_window_generator(self, output_config):

        train_window_gen = EnhancedWindowGenerator(
            mode='train',
            input_width=self.model_config['input_width'],
            label_width=self.model_config['output_width'],
            shift=self.model_config['shift'],
            label_columns=list(self.model_config['output_config'].keys()),
            numeric_columns=self.model_config['numeric_columns'],
            categorical_columns=self.model_config['categorical_columns'],
            embedding_configs=self.embedding_info_,
            output_configs=output_config
        )

        return train_window_gen

    def _forecast_window_generator(self):

        self.predict_window_gen = EnhancedWindowGenerator(
            mode='forecast',
            input_width=self.model_config['input_width'],
            # 预测模式不需要label_width和shift
            numeric_columns=self.model_config['numeric_columns'],
            categorical_columns=self.model_config.get('categorical_columns'),
            embedding_configs=self.embedding_info_,
        )

        self.forecast_window_config_ = {
            'mode': 'forecast',
            'input_width': self.model_config['input_width'],
            'numeric_columns': self.model_config['numeric_columns'],
            'categorical_columns': self.model_config.get('categorical_columns'),
            'embedding_configs': self.embedding_info_,

        }

        return self.predict_window_gen, self.forecast_window_config_



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

    # def __getstate__(self):
    #     """序列化时只保留必要信息"""
    #     state = self.__dict__.copy()
    #
    #     # 移除所有模型实例（通过save/load机制重建）
    #     state['training_model_'] = None
    #     state['prediction_model_'] = None
    #     state['window'] = None
    #
    #     return state
    #
    # def __setstate__(self, state):
    #     """反序列化"""
    #     self.__dict__.update(state)
    #
    #     if hasattr(self, 'weights_path') and os.path.exists(self.weights_path):
    #         self.prediction_model_ = self.load_best_model()
    #         self.train_window_gen_ = self._train_window_generator(self.model_config['output_config'])


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
                        serialized = cloudpickle.dumps(transformer)
                        serialized_states[pipe_name][step_name] = {
                            'pickled': serialized,
                            'type': type(transformer).__name__,
                            'pickle_type': 'cloudpickle',
                            'params': transformer.get_params() if hasattr(transformer, 'get_params') else {}
                        }
                        logger.debug(f"成功使用 cloudpickle 序列化 {pipe_name}.{step_name}")

                    except Exception as e:
                        logger.info(f"cloudpickle 序列化失败{pipe_name}.{step_name}:{e}")

                        # 尝试 fallback 到标准 pickle
                        try:
                            import pickle
                            serialized = pickle.dumps(transformer)
                            serialized_states[pipe_name][step_name] = {
                                'pickled': serialized,
                                'type': type(transformer).__name__,
                                'pickle_type': 'pickle',  # 标记使用的序列化方式
                                'params': transformer.get_params() if hasattr(transformer, 'get_params') else {}
                            }
                            logger.info(f"fallback: 使用标准 pickle 序列化 {pipe_name}.{step_name}")

                        except Exception as e2:
                            logger.error(f"所有序列化方法都失败 {pipe_name}.{step_name}: {e2}")

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
            'saved_at': datetime.datetime.now().isoformat()
        }
        with state_file.open('wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Pipeline状态已保存到: {state_file}")
        return save_data

    def custom_inverse_transform(self, raw_predictions: Dict, task_config, use_saved, output_width, **kwargs):
        """
        智能逆转换：根据情况选择使用内存引用或保存的状态
        支持多预测的逆转换

        Args:
           raw_predictions: 原始预测 字典格式，{task_name: prediction_array}
           use_saved: True=使用保存的状态，False=尝试使用内存引用
           task_config:output_config 获取任务类型
           **kwargs: 其他参数

        Returns:
           逆转换后的结果 _
           再判断是否有果power transformer
        """
        if not isinstance(raw_predictions, dict):
            raise TypeError(f"期望字典格式，但得到: {type(raw_predictions)}")

        logger.debug(f"[INFO] 多任务预测: {len(raw_predictions)}个任务")

        processed_tasks = {}

        for task_name, task_pred in raw_predictions.items():
            if isinstance(task_pred, tf.Tensor):
                task_pred = task_pred.numpy()

            logger.debug(f"[INFO] 任务{task_name}原始形状: {task_pred.shape}")
            task_type = task_config.get(task_name).get('type', 'regression')

            """
            多步预测:
            1.回归 (batch,output_width,1) ；二分类(batch,output_width,1)->可压缩->2D
            2.多分类(batch,output_width,num_classes) -> 不压缩->保持3D（ argmax ->2D)

            单步预测：
            1.回归（batch,1) ；二分类(batch,1)？->不压缩->2D
            2.多分类(batch,1,num_classes) -> 不压缩->保持3D（argmax->2D)

            逆标准化器：保证接受2D；
            逆编码器：保证接受3D，内部转换2D操作；
            """
            if output_width > 1:
                if task_type in ['regression', 'binary_classification']:
                    if task_pred.shape[-1] == 1:  # 去除冗余的最后一个维度(samples, output_width,1)
                        task_pred = np.squeeze(task_pred, axis=-1)
                        logger.debug(f"[INFO] regression任务{task_name}去除冗余后: {task_pred.shape}")

            if not use_saved and hasattr(self, '_temp_preprocessor'):
                task_result = self._inverse_transform_live(prediction=task_pred, target_column=task_name,
                                                           task_type=task_type, **kwargs)
            else:
                task_result = self._inverse_transform_from_saved(prediction=task_pred, target_column=task_name,
                                                                 task_type=task_type, **kwargs)

            logger.debug(f"[INFO] 任务{task_name}逆转换后: {task_result.shape}")
            processed_tasks[task_name] = task_result

        return processed_tasks

    def _inverse_transform_live(self, prediction: np.ndarray, pipeline_name='pipeline_6',
                                scale_step_names=None, target_column: str = None, task_type: str = None,
                                transform_step_name=None) -> np.ndarray:

        logger.debug(f"[DEBUG] _inverse_transform_live 开始")
        logger.debug(f"target_column: {target_column}")
        logger.debug(f"task_pred: {prediction.shape}")
        logger.debug(f"pipeline_name: {pipeline_name}")
        logger.debug(f"scale_step_names: {scale_step_names}")

        if scale_step_names is None:
            scale_step_names = ['engineer_3', 'engineer_4']

        if transform_step_name is None:
            transform_step_name = 'engineer_2'

        result = prediction

        for step_name in scale_step_names:
            transformer = self._temp_preprocessor.pipelines_[pipeline_name].named_steps[step_name]

            # 逆标准化
            if step_name == 'engineer_3':
                valid_col = transformer.without_outlier_missing_columns_

                # 普通数值列（非二分类列：特征/标记）
                if target_column is not None and task_type == 'regression' and target_column in valid_col:  # 只有数值列才进行标准化
                    result = transformer.custom_inverse_transform(scaled_data=result,
                                                                  target_column=target_column)  # 更新result

                # 数值二分类列 阈值管理
                elif target_column is not None and task_type == 'binary_classification' and target_column not in valid_col:
                    threshold = 0.5
                    result = (result > threshold).astype(int)

                else:
                    logger.debug(f"目标列{target_column}不需要数值列的逆标准化转换或者二分阈值管理")

            # 逆编码
            elif step_name == 'engineer_4':
                valid_col = transformer.categorical_columns_

                # 多分类 概率数组: (batch, timesteps, num_classes)
                if target_column is not None and task_type == 'classification' and target_column in valid_col:  # 只有多分类列才进行编码
                    result = transformer.custom_inverse_transform(scaled_data=result, target_column=target_column)
                else:
                    logger.debug(f"目标列{target_column}不需要分类列的逆编码转换")

        """标准化/编码后，再进行其他逆转换"""
        other_transformer = self._temp_preprocessor.pipelines_[pipeline_name].named_steps[transform_step_name]

        if target_column is not None:
            if target_column in other_transformer.valid_asinh_columns_:
                result = other_transformer.custom_inverse_transform(transformed_data=result,
                                                                    target_column=target_column, transform_type='asinh')

            # PowerTransformer 逆转换单列时需要模拟原始列数，但只填充目标列。
            elif target_column in other_transformer.valid_power_columns_:  # batch, output_width
                result = other_transformer.custom_inverse_transform(transformed_data=result,
                                                                    target_column=target_column, transform_type='power')

            else:
                logger.debug(f"目标列{target_column}不需要powertransform/asinh等逆转换")

        else:
            logger.debug(f"目标列{target_column}为空")

        return result

    def _inverse_transform_from_saved(self, prediction: np.ndarray, pipeline_name='pipeline_6',
                                      scale_step_names=None, target_column=None, task_type: str = None,
                                      transform_step_name=None) -> np.ndarray:

        if scale_step_names is None:
            scale_step_names = ['engineer_3', 'engineer_4']

        if transform_step_name is None:
            transform_step_name = ['engineer_2']

        result = prediction

        for step_name in scale_step_names:
            if pipeline_name in self.serialized_states and step_name in self.serialized_states[pipeline_name]:
                state_info = self.serialized_states[pipeline_name][step_name]

                # 从pickle重建transformer
                if state_info.get('pickled'):
                    transformer = pickle.loads(state_info['pickled'])

                    if hasattr(transformer, 'custom_inverse_transform'):
                        if step_name == 'engineer_3':
                            valid_col = transformer.without_outlier_missing_columns_
                            if target_column is not None and task_type == 'regression' and target_column in valid_col:
                                result = transformer.custom_inverse_transform(scaled_data=result,
                                                                              target_column=target_column)

                            elif target_column is not None and task_type == 'binary_classification' and target_column not in valid_col:
                                threshold = 0.5
                                result = (result > threshold).astype(int)

                            else:
                                logger.debug(f"目标列{target_column}不需要数值列的逆标准化转换或者二分阈值管理")
                        else:
                            valid_col = transformer.categorical_columns_
                            if target_column is not None and task_type == 'classification' and target_column in valid_col:
                                result = transformer.custom_inverse_transform(scaled_data=result,
                                                                              target_column=target_column)
                            else:
                                logger.debug(f"目标列{target_column}不需要分类列的逆编码转换")
                else:
                    logger.debug(f"pickled失败需要手动")

        """标准化/编码后，再进行其他逆转换"""
        other_transformer = self._temp_preprocessor.pipelines_[pipeline_name].named_steps[transform_step_name]

        if target_column is not None:
            if target_column in other_transformer.valid_asinh_columns_:
                result = other_transformer.custom_inverse_transform(transformed_data=result,
                                                                    target_column=target_column, transform_type='asinh')

            # PowerTransformer 逆转换单列时需要模拟原始列数，但只填充目标列。
            elif target_column in other_transformer.valid_power_columns_:  # batch, output_width
                result = other_transformer.custom_inverse_transform(transformed_data=result,
                                                                    target_column=target_column, transform_type='power')

            else:
                logger.debug(f"目标列{target_column}不需要powertransform/asinh等逆转换")

        else:
            logger.debug(f"目标列{target_column}为空")

        logger.debug(f"[DEBUG] _inverse_transform_saved 结束，返回类型: {type(result)}")
        return result

    def add_timestamps(self, predictions: Dict, historical_timestamps, input_width: int, output_width: int, shift: int,
                       freq: str):
        """
        参数:
            predictions:接收 逆转换处理过的predictions字典 ，值是array
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
        n_windows = len(historical_timestamps) - (input_width + shift + output_width - 1) + 1

        window_start_times = []  # 每个窗口的基准时间
        future_timestamps = []  # 每个窗口的预测时间点列表

        for i in range(n_windows):  # 预测不需要有真实值的窗口 最后1个i位置：len-input+1-1
            last_time = historical_timestamps.iloc[i + input_width - 1]  # 输入窗口的最后一条
            window_start_times.append(last_time)

            # 从base_time + shift(时间步）开始预测
            future_time = self._generate_future_timestamps(last_time,
                                                           n_steps=self.config.get('output_width', 1),
                                                           freq=freq,
                                                           shift=shift)
            future_timestamps.append(future_time)

        logger.debug("window_start_times 验证:")
        logger.debug(f"长度: {len(window_start_times)}")
        logger.debug(f"第一个: {window_start_times[0]}")
        logger.debug(f"最后一个: {window_start_times[-1]}")
        logger.debug(f"应该是: ({historical_timestamps.iloc[-5]} - 24h)")

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

    def _create_result_df(self, predictions, window_start_times: list, future_timestamps: list, ):
        """单任务和多任务区分（单：1个数组，多：每个元素是一个任务的输出
            df三列：开始时间、预测时间列、任务1，任务2"""

        task_names = self.config.get('task_names')

        num_windows = len(predictions[task_names[0]])
        logger.debug(num_windows)

        all_windows = []
        for i in range(num_windows):  # 窗口数量
            start_times = window_start_times[i]
            future_times = future_timestamps[i]

            for step in range(self.config.get('output_width', 1)):
                window = {
                    'window_end': start_times,
                    'forecast_time': future_times[step]}
                window.update(
                    **{f'{task_name}_pred': pred_values[i][step] for task_name, pred_values in
                       predictions.items()}  # 窗口定位 i
                )

                all_windows.append(window)

        logger.info(all_windows[-1])

        results_df = pd.DataFrame(all_windows)  # pd.concat 是组合df的，但这里是字典

        logger.debug(f"生成的预测记录总数: {len(results_df)}")  # 6980×5=34900
        logger.debug(f"CSV文件预览:")
        logger.debug(results_df.tail(10))

        results_df.to_csv(
            '/Users/shibo/Python/NeuralNetwork/temperature_forecasting/data/intermediate/predictions_result.csv',
            index=False)

        return results_df, predictions

    def calculate_mape(self, pred_data: pd.DataFrame, original_data: pd.DataFrame):

        time_col_name = self.config.get('time_col_name')
        task_names = self.config.get('task_names')

        selected_columns = [time_col_name] + [tk for tk in task_names]
        actual_data = original_data[selected_columns]

        combined = pd.merge(
            pred_data,
            actual_data,
            left_on='forecast_time',
            right_on=time_col_name,
            how='left',
        )
        logger.debug(f"合并后的数据是{combined.tail(10)}")

        # 逐时间步
        step_res = MetricsCalculator.calc_every_pair(data=combined, task_names=task_names)

        return step_res


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


class MetricsCalculator:
    """
        MAPE计算：每个时间点先平均预测值，再算一个APE
        MAE/MSE计算：应该用所有预测-实际对直接算
    """

    @staticmethod
    def calc_every_pair(data: pd.DataFrame, task_names: List[str]):
        """温度可能为负，分母也需要是绝对值"""

        for tk in task_names:
            mask = data[tk].notna() & (data[tk] != 0)
            data[f'ape_{tk}'] = np.nan
            data.loc[mask, f'ape_{tk}'] = np.round(
                np.abs(data.loc[mask, f'{tk}_pred'] - data.loc[mask, tk]) / np.abs(data.loc[mask, tk]) * 100, 4)

        return data

    @staticmethod
    def predictions_by_time(predictions, actual_data: pd.DataFrame, input_width: int, shift: int,
                            time_column: str = None,
                            level: str = 'o'):
        """""""""
         mape 先处理统计对，再计算
        ---------------------------

        参数：
        predictions ： 需要经过逆转换处理 {task_name: predictions}的格式 Mape / 或者未逆标准化的MAE MSE计算。每个任务有滑动窗口按需处理
        actual_data：原始的数据取对应的时间列 + 任务列，
        level: ‘D’ 代表 日级别 ,‘O' 单时间点级别，‘M’月级别
        shift:预测偏移

        某1个任务的示例：
        predictions: 每个窗口的预测结果列表 （数字是对应着输出结果timepoint的索引位置，理解 target_time ）
        如: [[105, 108, 112,111,222],  # 从t0,t1,t2,t3,t4,t5 预测 t29,t30,t31,t32,t33
            [112, 115, 118,111,111],  #  从t1,t2,t3,t4,t5,t6 预测 t30,t31,t32,t33,t34
            ...]
        t1预测两次 target_idx = window_start + steps_ahead + (input_width + shift - 1)
        actuals: 实际值列表 [100, 110, 105, 120, ...]
        """

        # 处理时间列
        if time_column is not None:
            if time_column in actual_data.columns:
                historical_timestamps = actual_data[time_column]
            else:
                warnings.warn(f"提供的time_column不在真实数据中")
                historical_timestamps = None
        else:
            time_column = actual_data.select_dtypes(include=['datetime64', np.datetime64, 'datetime']).columns.tolist()
            time_column = time_column[0]
            historical_timestamps = actual_data[time_column]

        # 处理对应关系
        result = {}
        for tk, prediction in predictions.items():
            predictions_by_time = defaultdict(list)

            for window_start, pred_window in enumerate(prediction):
                for steps_ahead, pred_value in enumerate(pred_window):
                    target_idx = window_start + steps_ahead + (input_width + shift - 1)  # 预测的目标时间点

                    time_point = historical_timestamps[target_idx]
                    if target_idx < len(historical_timestamps):
                        predictions_by_time[time_point].append(pred_value)  # value是表

            result[tk] = predictions_by_time

        return result

    @staticmethod
    def mape_calculator(predictions: Dict, actual_data: pd.DataFrame, input_width: int, shift: int,
                        time_column: str = None,
                        level: str = 'o'):
        """""""""      
        A.业务指标MAPE：
            逆标准化后的 predictions（predictions_by_time 字典）
            单时间点 mape + 整体 mape + 日级别 mape
            1. 单时间点：某时间点在预测中出现多次，先求预测的均值，再和actual_data的时间点进行计算；
            2. 日级别：聚合某日内所有预测时间点，再整理actual_data的日级别真实值，求ape，MAPE为同级别所有ape的均值mean(ape);
        """

        predictions_by_time = MetricsCalculator.predictions_by_time(predictions=predictions, actual_data=actual_data,
                                                                    input_width=input_width,
                                                                    shift=shift, time_column=time_column, level=level)
        result_mape = {}
        for tk, pred in predictions_by_time.items():
            mape_ = MetricsCalculator._calc_hierarchical_mape(pred, actual_data, level, tk, time_column)
            result_mape[tk] = mape_

        return result_mape

    @staticmethod
    def mae_mse_calculator(predictions: Dict, actual_data: pd.DataFrame, input_width: int, shift: int,
                           time_column: str = None,
                           level: str = 'o'):
        """""""""
        B.技术指标 MSE MAE：
          仍然是[标准化]的 predictions_by_time 字典
          最基本的计算单元是每个预测值与其对应实际值的比较(标准化状态下的）。无论什么层级，最终都是这些基础对的统计。
        """

        predictions_by_time = MetricsCalculator.predictions_by_time(predictions=predictions, actual_data=actual_data,
                                                                    input_width=input_width,
                                                                    shift=shift, time_column=time_column, level=level)

        result_mae_mse = {}
        for tk, pred in predictions_by_time.items():  # pairs 是时间点明细
            MAE, MSE, RMSE, pairs, mae_mse_ = MetricsCalculator._calc_hierarchical_mae_mse(pred,
                                                                                           actual_data,
                                                                                           level, tk)
            logger.info(f'任务{tk}整体数据的MAE：{MAE},MSE:{MSE},RMSE:{RMSE}')

            result_mae_mse[tk] = mae_mse_

        return result_mae_mse

    @staticmethod
    def download_data(result_mae_mse: Dict, result_mape: Dict, level_mae_mse: str, level_mape: str):
        """1个任务一个工作簿，1个指标占1个sheet"""
        if level_mae_mse.lower() != level_mape.lower():
            logger.warning(f"mae_mse 计算指标维度不统一")

        for tk, result in result_mae_mse.items():
            file = f'/Users/shibo/Python/NeuralNetwork/temperature_forecasting/data/intermediate/{tk}_metrics.xlsx'

            if tk in result_mape.keys():
                _mape = result_mape.get(tk)
                mape = _mape.get(f'details_{level_mape}')
                mae_mse = result

                sheet_data = {
                    'mape': mape,
                    'mae_mse': mae_mse
                }

                with pd.ExcelWriter(file, engine='openpyxl') as writer:
                    for sheet_name, df in sheet_data.items():
                        df.to_excel(writer, sheet_name=sheet_name)

    @staticmethod
    def _calc_hierarchical_mape(predictions_by_time: Dict, actual_data: pd.DataFrame, level: str, tk: str,
                                time_column: str):

        # 3.1 mape: add 时间点维度
        if level.lower() == 'o':
            avg_timepoint_predictions = []
            for time_point, preds in predictions_by_time.items():
                avg_timepoint_predictions.append(
                    {'timestamp': time_point,
                     f'{tk}_pred': np.mean(preds)
                     }
                )
            avg_predictions_df = pd.DataFrame(avg_timepoint_predictions).set_index('timestamp')
            timepoint_actual_data = actual_data.set_index(time_column)

            timepoint_mape, timepoint_details = MetricsCalculator.calc_mape(avg_predictions_df,
                                                                            timepoint_actual_data[tk])
            timepoint_analyze = MetricsCalculator.analyze_mape(details=timepoint_details)

            result = {
                f'details_{level}': timepoint_details,
                f'mape_analyze_{level}':
                    {'patterns': timepoint_analyze[0],
                     'consecutive': timepoint_analyze[1],
                     'weaknesses': timepoint_analyze[2]}}

        # 3.2 mape:add 各级别
        else:
            avg_predictions = []

            if level.lower() == 'd':
                for timestamp, preds in predictions_by_time.items():
                    timestamp = pd.Timestamp(timestamp.strftime('%Y-%m-%d'))
                    avg_predictions.append(
                        {'timestamp': timestamp,
                         f'{tk}_pred': np.mean(preds)})

                level_actual_data = (actual_data.assign(
                    level=actual_data[time_column].values.astype('datetime64[D]')
                ).groupby('level').mean(numeric_only=True))

            elif level.lower() == 'm':
                for timestamp, preds in predictions_by_time.items():
                    timestamp = pd.to_datetime(timestamp.strftime('%Y-%m'), format='%Y-%m')
                    avg_predictions.append(
                        {'timestamp': timestamp,
                         f'{tk}_pred': np.mean(preds)})

                level_actual_data = (actual_data.assign(
                    level=actual_data[time_column].values.astype('datetime64[M]')
                ).groupby('level').mean(numeric_only=True))

            else:
                for timestamp, preds in predictions_by_time.items():
                    timestamp = pd.Timestamp(timestamp.strftime('%Y'))
                    avg_predictions.append(
                        {'timestamp': timestamp,
                         f'{tk}_pred': np.mean(preds)})

                level_actual_data = (actual_data.assign(
                    level=actual_data[time_column].values.astype('datetime64[Y]')
                ).groupby('level').mean(numeric_only=True))

            avg_predictions_df = pd.DataFrame(avg_predictions)
            level_prediction_data = avg_predictions_df.groupby('timestamp').mean(numeric_only=True)

            level_mape, level_details = MetricsCalculator.calc_mape(level_prediction_data, level_actual_data[tk])
            daily_analyze = MetricsCalculator.analyze_mape(details=level_details)

            logger.info(f"该{level}级别的mape:{level_mape}")
            result = {
                f'details_{level}': level_details,
                f'mape_analyze_{level}':
                    {'patterns': daily_analyze[0],
                     'consecutive': daily_analyze[1],
                     'weaknesses': daily_analyze[2]}

            }

        return result

    @staticmethod
    def calc_mape(handled_pred: pd.DataFrame, handled_actual: pd.Series):
        """
        参数：
        pred_dict： 转换的时间维度 的预测值的均值（已处理）
        handled_actual：转换的时间维度 的真实值（已处理），actual 包含1列tk(任务列），索引是level
        接收单任务
        """

        detailed_results = pd.DataFrame()

        avg_pred = handled_pred.iloc[:, 0].values

        actual = handled_actual.values[-len(avg_pred):]  # 截掉开头非预测时间点

        detailed_results['timestamp'] = handled_pred.index
        detailed_results['actual'] = actual
        detailed_results['avg_pred'] = avg_pred

        mask = ((~np.isnan(actual) & ~np.isnan(avg_pred)) &
                (~np.isinf(actual) & ~np.isinf(avg_pred)) &
                (actual != 0))

        detailed_results['level_ape'] = np.nan

        ape_values = np.abs(actual[mask] - avg_pred[mask]) / np.abs(actual[mask]) * 100
        detailed_results.loc[mask, 'level_ape'] = ape_values

        # 计算有效ape的平均，不能先求和再计算
        if len(detailed_results) > 0:
            mape = np.mean(ape_values)  # 该级别
        else:
            mape = float('nan')

        return mape, detailed_results

    @staticmethod
    def analyze_mape(details: pd.DataFrame):
        """接收单任务"""

        patterns = {}
        consecutive_errors = []
        weaknesses = {}

        # 系统性错误评估(ape)
        # details = details.copy()
        # details['timestamp'] = pd.to_datetime(details['timestamp'],errors='coerce')
        # details = details.dropna(subset=['timestamp'])

        high_mask = details['level_ape'] >= 50

        if high_mask.any():
            outliers = details[high_mask].copy()

            if not outliers.empty:
                hour_counts = outliers['timestamp'].apply(lambda x: x.hour).value_counts().to_dict()
                month_counts = outliers['timestamp'].apply(lambda x: x.month).value_counts().to_dict()
                year_counts = outliers['timestamp'].apply(lambda x: x.year).value_counts().to_dict()

                patterns = {
                    'count': int(high_mask.sum()),
                    'error_rate': f'{high_mask.mean():.4f}',
                    'high_error_hours': hour_counts,
                    # 必须保证level为'o';outliers里没有nan 直接总行数即可;
                    'high_error_month': month_counts,
                    'high_error_year': year_counts,

                    'actual_range': {
                        'min_actual_in_outliers': outliers['actual'].min(),
                        'max_actual_in_outliers': outliers['actual'].max(),
                        'mean_actual_in_outliers': outliers['actual'].mean(),
                    }}

            # 连续错误(ape)
            consecutive_errors = []
            timestamps = details['timestamp']
            i = 0
            while i < len(high_mask):
                if high_mask[i]:
                    start = i
                    while i < len(high_mask) and high_mask[i]:
                        i += 1
                    end = i - 1

                    if end - start + 1 >= 3:  # 至少连续3个错误点
                        consecutive_errors.append(
                            {'start_time': timestamps[start],
                             'end_time': timestamps[end],
                             'duration_hours': f"{(timestamps[end] - timestamps[start]).total_seconds() / 3600}h"
                             })
                else:
                    i += 1

        # （actual)突变点、突变点位置的整体mape
        actual = details['actual'].values
        predicted = details['avg_pred'].values

        actual_change = np.abs(np.diff(actual, prepend=actual[0]))  # 在最前面1个数的加上actual[0], 加上1个diff = 0 ，保持长度=原长
        valid_mask = ~np.isnan(actual_change)

        if valid_mask.any() > 0:
            valid_changes = actual_change[valid_mask]
            threshold = np.mean(valid_changes) + 2 * np.std(valid_changes)

            spike_mask = (actual != 0) & (actual_change > threshold)
            if spike_mask.any():
                details['spike'] = spike_mask.astype(int)
                weaknesses = {
                    'spike_points_mape': np.abs(actual[spike_mask] - predicted[spike_mask]) / np.abs(
                        actual[spike_mask]) * 100,
                    'spike_count': spike_mask.sum()}

        return patterns, consecutive_errors, weaknesses

    @staticmethod
    def _calc_hierarchical_mae_mse(predictions_by_time: Dict, actual_data, level: str, tk: str):
        """ 标准化数据
        predictions_by_time: {时间点索引: [该时间点的所有预测值]}
        actual_data: 原数据DF，每个时间点的实际值
        所有预测-实际对 pairs 直接计算
        """
        actual_data = actual_data[tk].values
        pairs_res = []  # 所有样本(pairs)带有时间点的计算结果

        for i, (time_point, pred_list) in enumerate(predictions_by_time.items()):
            if i < len(actual_data):
                actual = actual_data[i]

                for pred in pred_list:
                    abs_error = abs(pred - actual)
                    squared_error = (pred - actual) ** 2
                    pairs_res.append(
                        {'timepoint': time_point,
                         'abs_error': abs_error,
                         'squared_error': squared_error,
                         })

        pairs_df = pd.DataFrame(pairs_res)

        if level.lower() == 'o':
            pairs_df = pairs_df.rename(columns={'timepoint': 'level'})
        elif level.lower() == 'd':
            pairs_df['level'] = pairs_df['timepoint'].values.astype('datetime64[D]')
        elif level.lower() == 'm':
            pairs_df['level'] = pairs_df['timepoint'].values.astype('datetime64[M]')
        else:
            pairs_df['level'] = pairs_df['timepoint'].values.astype('datetime64[Y]')

        hierarchical_metrics = pairs_df.groupby('level').agg(mae=('abs_error', 'mean'),  # mae
                                                             mse=('squared_error', 'mean'),
                                                             rmse=('squared_error', lambda x: np.sqrt(x.mean())  # rmse
                                                                   ))
        MAE = pairs_df['abs_error'].mean()
        MSE = pairs_df['squared_error'].mean()
        RMSE = np.sqrt(MSE)
        return MAE, MSE, RMSE, pairs_df, hierarchical_metrics


if __name__ == '__main__':
    import pandas as pd

    ## 示例1：使用字符串时间键
    dates = ['2016-12-31 17:00:00', '2016-12-31 18:00:00', '2016-12-31 19:00:00', '2016-12-31 20:00:00',
             '2016-12-31 21:00:00', '2016-12-31 22:00:00', '2016-12-31 23:00:00', '2017-01-01 00:00:00']

    no_scaled_data = {
        'Date Time': ['2016-12-31 17:00:00', '2016-12-31 18:00:00', '2016-12-31 19:00:00', '2016-12-31 20:00:00',
                      '2016-12-31 21:00:00', '2016-12-31 22:00:00', '2016-12-31 23:00:00', '2017-01-01 00:00:00'],
        'T': [1.41, -0.08, -1.03, -1.52, -3.09, -2.59, -3.76, -4.82],  #
        'rh': [64.81, 69.81, 70.7, 65.42, 73.7, 71.3, 72.5, 75.7]
    }
    no_scaled_data = pd.DataFrame(no_scaled_data)
    no_scaled_data['Date Time'] = pd.to_datetime(no_scaled_data['Date Time'], format='%Y-%m-%d %H:%M:%S')
    predictions_df = {
        'window_end': ['2016-12-31 17:00:00', '2016-12-31 17:00:00', '2016-12-31 17:00:00', '2016-12-31 17:00:00',
                       '2016-12-31 17:00:00',
                       '2016-12-31 18:00:00', '2016-12-31 18:00:00', '2016-12-31 18:00:00', '2016-12-31 18:00:00',
                       '2016-12-31 18:00:00',
                       '2016-12-31 19:00:00', '2016-12-31 19:00:00', '2016-12-31 19:00:00', '2016-12-31 19:00:00',
                       '2016-12-31 19:00:00',
                       '2016-12-31 20:00:00', '2016-12-31 20:00:00', '2016-12-31 20:00:00', '2016-12-31 20:00:00',
                       '2016-12-31 20:00:00'],
        'forecast_time': ['2016-12-31 17:00:00', '2016-12-31 18:00:00', '2016-12-31 19:00:00', '2016-12-31 20:00:00',
                          '2016-12-31 21:00:00',
                          '2016-12-31 18:00:00', '2016-12-31 19:00:00', '2016-12-31 20:00:00', '2016-12-31 21:00:00',
                          '2016-12-31 22:00:00',
                          '2016-12-31 19:00:00', '2016-12-31 20:00:00', '2016-12-31 21:00:00', '2016-12-31 22:00:00',
                          '2016-12-31 23:00:00',
                          '2016-12-31 20:00:00', '2016-12-31 21:00:00', '2016-12-31 22:00:00', '2016-12-31 23:00:00',
                          '2017-01-01 00:00:00'],
        'T_pred': [3.8703365, 3.884691, 3.4577994, 3.7306015, 2.2956214, 2.6391318, 2.5926297, 2.2178895, 2.5358593,
                   1.0858217, 1.77491, 1.7106596, 1.1285466, 1.1749766, -0.014756217, 0.88609976, 0.75710475,
                   0.19265927, 0.11487619, -0.86728024],
        'rh_pred': [79.67318, 80.484695, 80.43478, 83.38382, 83.45128, 80.5278, 81.90515, 81.87326, 85.21435, 84.81238,
                    81.605484, 83.30215, 83.313484, 86.7817, 86.10873, 83.98053, 85.9154, 85.84884, 89.42158, 88.54782]
    }

    predictions_df = pd.DataFrame(predictions_df)
    predictions_df['window_end'] = pd.to_datetime(predictions_df['window_end'], format='%Y-%m-%d %H:%M:%S')
    predictions_df['forecast_time'] = pd.to_datetime(predictions_df['forecast_time'], format='%Y-%m-%d %H:%M:%S')

    predictions = {
        'T': [
            [3.8703365, 3.884691, 3.4577994, 3.7306015, 2.2956214],
            [2.6391318, 2.5926297, 2.2178895, 2.5358593, 1.0858217],
            [1.77491, 1.7106596, 1.1285466, 1.1749766, -0.014756217],
            [0.88609976, 0.75710475, 0.19265927, 0.11487619, -0.86728024]
        ],
        'rh': [[79.67318, 80.484695, 80.43478, 83.38382, 83.45128],
               [80.5278, 81.90515, 81.87326, 85.21435, 84.81238, ],
               [81.605484, 83.30215, 83.313484, 86.7817, 86.10873],
               [83.98053, 85.9154, 85.84884, 89.42158, 88.54782]]
    }

    # 验证生成基础dataframe step calculate_mape
    time_column = 'Date Time'
    task_names = ['T', 'rh']
    combined = pd.merge(
        predictions_df.copy(),
        no_scaled_data.copy(),
        left_on='forecast_time',
        right_on=time_column,
        how='left',
    )
    logger.debug(f"合并后的数据是{combined.tail(10)}")

    # 验证逐时间步
    step_res = MetricsCalculator.calc_every_pair(data=combined, task_names=task_names)
    print(step_res.head(50))

    # 逐时间点timepoint / 整体mape / 日级别 mape
    mape_dict = MetricsCalculator.predictions_by_time(predictions=predictions, actual_data=no_scaled_data.copy(),
                                                      input_width=6, shift=24, level='o')
    logger.debug(f"mape_dict含有的任务：{mape_dict.keys}")
