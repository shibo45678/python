import random
import cloudpickle
from datetime import datetime
import os
from typing import Dict, Any
import re
import datetime
import numpy as np
from pydantic.v1 import validate_arguments
from pydantic import Field
from sklearn.utils.validation import check_is_fitted
from data.decorator import validate_input
from evaluation.model_feature_importance import FeatureImportance
from models.cnn import MultiTasksCnnModel
from models.lstm import SingleTaskLstmModel, MultiTasksLstmModel
from training.training_models import TrainingSingleModel, TrainingMultiModel

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
        self.input_cols_ = None  # 处理掉时间列，保证进入模型的所有列是数值
        self.train_window_gen_ = None
        self.forecast_window_gen_ = None
        self.forecast_window_config_ = None
        self.stage_number_ = -1

    def fit(self, X: dict, y=None):
        train_data = X.get('train_data')
        val_data = X.get('val_data')

        # 0.处理时间列
        datetime_cols = train_data.select_dtypes(include=['datetime64']).columns
        self.input_cols_ = [col for col in list(train_data.columns) if col not in datetime_cols]

        train_datasets_ = train_data[self.input_cols_]
        val_datasets_ = val_data[self.input_cols_]

        #  构建模型 （神经网络预处理已经返回了模型期望的正确格式，不copy）
        # 1.1 获得embedding_info
        self.embedding_info_ = EmbeddingConfig._get_embedding_info(train_datasets_,  # 原始DF
                                                                   self.model_config['categorical_columns']
                                                                   )

        # 1.2 处理窗口数据
        self.train_window_gen_ = self._train_window_generator(self.model_config['output_config'])
        self.train_window_data_ = self.train_window_gen_.createDataset(train_datasets_)
        self.val_window_data_ = self.train_window_gen_.createDataset(val_datasets_)


        """
        在保证不影响 main函数，目前 preprocessor,postprocess 以及deployment逻辑（保证进程活着）的前提下，存储模型。self.best_checkpoint 来源：
        1. 训练：使用main.py进行首次或者继续训练，当次得到的最佳模型
        2. 直接加载已有模型：使用continue.py单独进行训练后的模型
        该位置需要保证：embedding_info_纳入
        """
        final_best_model = self.model_config.get('final_best_model', None)

        if final_best_model is None:

            # 1.3 训练
            continue_train = self.model_config.get('continue_from', None)
            model_name = self.model_config.get('model_type', 'unknown')
            multi_model = self.model_config.get('multi_tasks', False)

            training_config = {'model_name': model_name, 'continue_from': continue_train,
                               'trainset': self.train_window_data_, 'valset': self.val_window_data_,

                               'learning_rate': self.model_config['learning_rate'],
                               'total_epochs': self.model_config['total_epochs'],
                               'cos_min_lr': self.model_config['cos_min_lr'],
                               'cos_total_epochs': self.model_config['cos_total_epochs'],
                               'cos_warmup_epochs': self.model_config['cos_warmup_epochs'],
                               'verbose': self.model_config['verbose'],

                               'early_stop_patience': self.model_config['early_stop_patience'],
                               'min_delta': self.model_config['min_delta'],
                               'check_save_mode': self.model_config['check_save_mode'],
                               'gap_tolerance_ratio': self.model_config['gap_tolerance_ratio'],
                               'min_gap_threshold': self.model_config['min_gap_threshold'],
                               'output_config':self.model_config['output_config'],
                               'weight_decay':self.model_config['weight_decay'],
                               'clipnorm':self.model_config['clipnorm']
                               }

            if continue_train is None:
                # 首次（构建模型）
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                basic_dir = f"saved_model/{self.model_config['model_type']}_{timestamp}"
                os.makedirs(basic_dir, exist_ok=True)

                continue_training_dir = os.path.join(basic_dir, 'continue_training')
                os.makedirs(continue_training_dir, exist_ok=True)

                self.stage_number_ = -1

                self._save_preprocessed_data(continue_training_dir=continue_training_dir,
                                             trainset=self.train_window_data_,
                                             valset=self.val_window_data_)
            else:
                # 继续（直接加载）
                basic_dir = None
                match = re.search(r"(.*)tf_checkpoints_stage(\d+)", str(continue_train))
                continue_training_dir = os.path.join(match.group(1), 'continue_training')
                self.stage_number_ = int(match.group(2))

            training_config.update({'basic_dir': basic_dir})

            # 1.3.1 多任务/单任务分支
            if multi_model:

                if continue_train is not None:
                    first_training_config_multi = {**training_config,
                                                   'model': None,
                                                   'monitor': self.model_config['monitor']}
                else:
                    if model_name.startswith('multi_lstm'):
                        model_config = {**self.model_config,
                                        'embedding_configs': self.embedding_info_}

                        lstm_model = MultiTasksLstmModel(model_config)
                        lstm_model_ = lstm_model._build_lstm_model()
                        self.training_model_ = lstm_model_

                    elif model_name.startswith('multi_cnn'):
                        model_config = {**self.model_config,  # 解包
                                        'embedding_configs': self.embedding_info_}  # 追加
                        cnn_model = MultiTasksCnnModel(architecture_type='enhance_parallel', config=model_config)
                        cnn_model_ = cnn_model._build_cnn_model()
                        self.training_model_ = cnn_model_

                    else:
                        raise ValueError(f"未支持的模型{model_name}")

                    first_training_config_multi = {**training_config,
                                                   'model': self.training_model_,
                                                   'monitor': self.model_config['monitor']}

                train = TrainingMultiModel()
                self.history_, best_checkpoint_epoch = train.training_model(**first_training_config_multi)


            # 1.3.2 单任务
            else:
                if continue_train is not None:
                    first_training_config_single = {**training_config, 'model': None}
                else:
                    if model_name.startswith('single_lstm'):
                        model_config = {**self.model_config,
                                        'embedding_configs': self.embedding_info_}
                        lstm_model = SingleTaskLstmModel(model_config)
                        lstm_model_ = lstm_model._build_lstm_model()
                        self.training_model_ = lstm_model_
                    else:
                        raise ValueError(f"未支持的模型{self.model_config['model_type']}")

                    first_training_config_single = {**training_config, 'model': self.training_model_}

                train = TrainingSingleModel()
                self.history_, best_checkpoint_epoch = train.training_model(**first_training_config_single)

            #  存每次训练的配置 + 训练历史
            self._save_model_config(continue_training_dir=continue_training_dir, stage_number=self.stage_number_)

            self._save_training_history(history=self.history_, continue_training_dir=continue_training_dir,
                                        stage_number=self.stage_number_)


        # continue_training.py加载继续训练后的最佳模型
        else:
            best_checkpoint_epoch = final_best_model

        # 保存最佳检查点路径供后续使用
        self.best_checkpoint = best_checkpoint_epoch  # 带epoch/

        # 训练完成后，创建用于预测的模型
        self.prediction_model_ = self.load_best_model()

        # 1.5 评估模型
        self.evaluate_model(dataset=self.val_window_data_, dataset_type='val')

        # 1.6 计算验证集特征重要性
        computer = FeatureImportance()
        computer.permutation_importance_lstm(model=self.prediction_model_,valsets=self.val_window_data_,
                                             n_repeats=5,output_configs=self.model_config['output_config'],
                                             num_feature_names =self.model_config.get('numeric_columns'),
                                             cat_feature_names = self.model_config.get('categorical_columns'),
                                             model_name = self.model_config['model_type'])
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

    def permutation_importance_lstm(self, model, X_val, y_val, feature_names):
        """计算 LSTM 的排列重要性
            model: 训练好的 LSTM 模型
            X_val: 验证数据 (3D: [样本, 时间步, 特征])  验证集

            y_val: 验证标签             raw_predictions(字典格式）
            feature_names: 特征名称列表
            n_repeats: 每个特征重复打乱的次数"""

    def _save_preprocessed_data(self, continue_training_dir, trainset, valset):
        model_name = self.model_config['model_type']
        saved_data_path = os.path.join(continue_training_dir, f'{model_name}_preprocessed_data')
        os.makedirs(saved_data_path, exist_ok=True)

        train_save_path = os.path.join(saved_data_path, 'train_dataset')
        trainset.save(train_save_path)

        val_save_path = os.path.join(saved_data_path, 'val_dataset')
        valset.save(val_save_path)

    def _save_model_config(self, continue_training_dir, stage_number):
        model_name = self.model_config['model_type']
        saved_config_path = os.path.join(continue_training_dir, f'{model_name}_config_stage{stage_number + 1}.cpkl')
        with open(saved_config_path, 'wb') as f:
            cloudpickle.dump(self.model_config, f)

    def _save_training_history(self, history, continue_training_dir, stage_number):
        model_name = self.model_config.get('model_type', 'unknown')
        history_path = os.path.join(continue_training_dir, f'{model_name}_history_stage{stage_number + 1}.cpkl')
        csv_path = os.path.join(continue_training_dir, f'{model_name}_history_stage{stage_number + 1}.csv')
        # history 是一个 Keras History 对象 不可以直接dump
        # history.history：字典 / history.params：字典（可序列化） / history.epoch：列表（可序列化） 其他不可序列化

        if hasattr(history, 'history'):
            history_dict = history.history if hasattr(history, 'history') else history
            epochs = history.epoch if hasattr(history, 'epoch') else list(range(len(history_dict.get('loss', []))))
            params = history.params if hasattr(history, 'params') else {}
        else:
            history_dict = history
            epochs = list(range(1, len(next(iter(history_dict.values())))))
            params = {}

        history_data = {
            'model_name': model_name,
            'history': history_dict,
            'epochs': [int(e) for e in epochs],
            'params': params,
            'stage': stage_number + 1,
            'save_time': datetime.datetime.now().isoformat()
        }
        with open(history_path, 'wb') as f:
            cloudpickle.dump(history_data, f)

        df = pd.DataFrame(history_dict)
        df.insert(0, 'epoch', epochs)
        df.to_csv(csv_path, index=False)
        logger.info(f"训练历史保存到: {csv_path}")
        return history_path, csv_path

    def load_best_model(self):

        if not hasattr(self, 'best_checkpoint'):
            raise ValueError('未找到最佳模型检查点')

        checkpoint_dir = self.best_checkpoint
        file_list = os.listdir(checkpoint_dir)
        keras_files = []

        for file in file_list:
            if file.endswith('.keras'):
                keras_file = os.path.join(checkpoint_dir, file)
                keras_files.append(keras_file)

        if keras_files:
            keras_files.sort(key=os.path.getmtime, reverse=True)
            model = tf.keras.models.load_model(keras_files[0])
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
            output_configs=output_config,
            batch_size=self.model_config['batch_size']
        )

        return train_window_gen

    def _forecast_window_generator(self):

        self.forecast_window_gen_ = EnhancedWindowGenerator(
            mode='forecast',
            input_width=self.model_config['input_width'],
            # 预测模式不需要label_width和shift
            numeric_columns=self.model_config['numeric_columns'],
            categorical_columns=self.model_config.get('categorical_columns'),
            embedding_configs=self.embedding_info_,
            batch_size=self.model_config['batch_size']
        )

        self.forecast_window_config_ = {
            'mode': 'forecast',
            'input_width': self.model_config['input_width'],
            'numeric_columns': self.model_config['numeric_columns'],
            'categorical_columns': self.model_config.get('categorical_columns'),
            'embedding_configs': self.embedding_info_,
            'batch_size': self.model_config['batch_size']
        }

        return self.forecast_window_gen_, self.forecast_window_config_



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


