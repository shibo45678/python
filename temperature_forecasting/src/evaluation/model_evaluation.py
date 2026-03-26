import os

os.environ['PYTHON_THREAD'] = 'child'
import matplotlib

matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import numpy as np
from typing import Dict
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)
from src.data.windows import EnhancedWindowGenerator



class ModelEvaluation:
    def __init__(self, output_configs: Dict, model_name: str):
        self.output_configs = output_configs
        self.model_name = model_name
        self.task_order = list(output_configs.keys())

    def comprehensive_model_evaluation(self,
                                       model,  # tf.keras.Model 防误导
                                       window: 'EnhancedWindowGenerator',
                                       dataset: tf.data.Dataset,
                                       dataset_type: str
                                       ) -> Dict:
        # 1. 基础评估
        logger.debug("=" * 60)
        logger.debug(f"开始评估 {self.model_name}")
        logger.debug("=" * 60)

        task_metrics = self._evaluate_multi_task_model(model, window, dataset, dataset_type)

        data_analysis = self._detailed_multi_task_evaluation(model, dataset, dataset_type)

        return {
            'model_name': self.model_name,
            'task_metrics': task_metrics,  # 基础指标
            'dataset_analysis': data_analysis,  # 详细分析
        }


    def _evaluate_multi_task_model(self, model, window, dataset, dataset_type) -> Dict:
        """混合分类和回归"""
        """
        评估多任务模型（混合回归和分类）
        Args:
            model: 已训练的Keras模型
            window: 窗口生成器对象
        """

        # 绘制预测效果
        if hasattr(window, 'enhanced_window_plot'):
            window.enhanced_window_plot(model=model, model_name=self.model_name, dataset=dataset,
                                        save_path='~/Python/NeuralNetwork/temperature_forecasting/data/pics')  # 使用已训练好的模型，拿训练集的example直接预测看结果

        # 评估模型
        # 分别适配多任务和单任务
        data_metrics = model.evaluate(dataset, verbose=0, return_dict=True)  # 所有损失和指标的'数值'列表 使用return_dict=True

        logger.debug(f"=== {self.model_name} 模型评估结果 ===")

        # 为每个任务单独计算指标
        task_metrics = {}
        task_num = len(self.output_configs.keys())
        for task_name, config in self.output_configs.items():
            task_type = config['type']

            logger.debug(f"--- 任务: {task_name} ({task_type}) ---")
            metric_name1 = 'loss' if task_num == 1 else f'{task_name}_loss'
            data_loss = data_metrics.get(metric_name1, 0)
            logger.debug(f"{task_name} - 整体{dataset_type}-{metric_name1}: {data_loss:.4f}")

            if task_type == 'regression':
                metric_name2 = 'mae' if task_num == 1 else f'{task_name}_mae'
                data_metric = data_metrics.get(metric_name2, 0)
            else:  # binary_classification + 多分类
                metric_name2 = 'accuracy' if task_num == 1 else f'{task_name}_accuracy'
                data_metric = data_metrics.get(metric_name2, 0)
            logger.debug(f"{task_name} - 整体{dataset_type}-{metric_name2}: {data_metric:.4f}")

            # 存储任务指标
            task_metrics[task_name] = {
                f'{dataset_type}_{metric_name1}': data_loss,
                f'{dataset_type}_{metric_name2}': data_metric,
                'type': task_type
            }

        return task_metrics

    def _detailed_multi_task_evaluation(self,
                                        model: tf.keras.Model,
                                        dataset: tf.data.Dataset,
                                        dataset_type: str) -> Dict:

        # 获取一批数据进行详细分析
        inputs, true_labels = next(iter(dataset))  # (tuple ,dict)
        predictions = model.predict(inputs, verbose=0)  # dict

        logger.debug(f"=== {self.model_name} - {dataset_type} 详细分析（单批） ===")
        task_results = {}

        # 多输出模型：predictions是dict
        if isinstance(predictions, (tuple, dict)):
            if isinstance(inputs, tuple):
                logger.debug(
                    f"inputs的数值特征 (batch_size, sequence_length, total_features): {inputs[0].shape}")  # 分类特征不一定有，所以不写 (batch_size, sequence_length, total_features)

            if isinstance(true_labels, dict):
                logger.debug(f"true_labels特征数: {len(true_labels.keys())}")

            for i, (task_name, config) in enumerate(self.output_configs.items()):
                if i < len(predictions):
                    pred = predictions[task_name]   # 第i个输出层的预测
                    true = true_labels[task_name]  # 根据key提取对应value
                    task_results[task_name] = self._analyze_single_task(pred, true, config, task_name)

        return task_results

    def _analyze_single_task(self, predictions: np.ndarray,
                             true_values: np.ndarray,
                             config: Dict,
                             task_name: str) -> Dict:

        task_type = config['type']

        if task_type == 'regression':
            return self._analyze_regression_task(predictions, true_values, task_name)

        elif task_type == 'binary_classification':
            # 二分类分析
            return self._analyze_binary_classification_task(predictions, true_values, task_name)

        elif task_type == 'classification':
            # 多分类分析
            return self._analyze_multiclass_task(predictions, true_values, task_name)
        else:
            logger.debug(f"未知任务类型: {task_type}")
            return {}

    def _analyze_regression_task(self, predictions: np.ndarray,
                                 true_values: np.ndarray,
                                 task_name) -> Dict:

        # 确保数据形状正确
        predictions = tf.squeeze(predictions)  # ->(32,5)
        true_values = tf.squeeze(true_values)  # (32,5,1) - > (32,5) 多余的1维拿掉

        mae = np.mean(np.abs(predictions - true_values))
        mse = np.mean((predictions - true_values) ** 2)
        rmse = np.sqrt(mse)

        logger.debug(f"-----{task_name}-----")
        logger.debug(f"MAE: {mae:.4f}")
        logger.debug(f"MSE: {mse:.4f}")
        logger.debug(f"RMSE: {rmse:.4f}")

        # 使用 TensorFlow 函数获取最小最大值
        pred_min = tf.reduce_min(predictions).numpy()
        pred_max = tf.reduce_max(predictions).numpy()
        true_min = tf.reduce_min(true_values).numpy()
        true_max = tf.reduce_max(true_values).numpy()

        logger.debug(f"预测值范围: [{pred_min:.3f}, {pred_max:.3f}]")
        logger.debug(f"真实值范围: [{true_min:.3f}, {true_max:.3f}]")
        logger.debug(f"预测值形状: {predictions.shape}")
        logger.debug(f"真实值形状: {true_values.shape}")

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'predictions': predictions,
            'true_values': true_values
        }

    def _analyze_binary_classification_task(self, predictions: np.ndarray,
                                            true_values: np.ndarray,
                                            task_name: str
                                            ) -> Dict:
        """分析二分类任务"""
        pred_probs = tf.squeeze(predictions)  # 概率值
        true_binary = tf.squeeze(true_values).astype(int)

        pred_binary = (pred_probs > 0.5).astype(int)
        accuracy = np.mean(pred_binary == true_binary)  # 准确率 = 正确预测的样本数 / 总样本数

        logger.debug(f"-----{task_name}-----")
        logger.debug(f"Accuracy: {accuracy:.4f}")
        logger.debug("分类报告:")
        logger.debug(classification_report(true_binary, pred_binary, zero_division=0))

        # 混淆矩阵
        cm = confusion_matrix(true_binary, pred_binary)
        logger.debug("混淆矩阵:")
        logger.debug(cm)

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': pred_binary,
            'probabilities': pred_probs,
            'true_values': true_binary
        }

    def _analyze_multiclass_task(self, predictions: np.ndarray,
                                 true_values: np.ndarray,
                                 task_name: str
                                 ) -> Dict:

        """分析多分类任务"""

        pred_probs = predictions
        pred_classes = np.argmax(predictions, axis=-1)  # (batch,) batch_size, output_width
        # np.argmax() 返回数组中最大值的索引 ,每个样本中最大概率的索引
        # 样本1: max(0.1, 0.8, 0.1) = 0.8 → 索引1 _>缩成数值类似格式
        # 样本2: max(0.7, 0.2, 0.1) = 0.7 → 索引0

        true_classes = tf.squeeze(true_values).astype(int)  # (batch, 1, 1) 移除数组中维度为1的轴。 (32,5)
        accuracy = np.mean(pred_classes == true_classes)  # 变成同样的1维数组比较

        logger.debug(f"-----{task_name}-----")
        logger.debug(f"Accuracy: {accuracy:.4f}")
        logger.debug("分类报告:")
        logger.debug(classification_report(true_classes, pred_classes, zero_division=0))

        # 混淆矩阵
        cm = confusion_matrix(true_classes, pred_classes)
        logger.debug("混淆矩阵:")
        logger.debug(cm)

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': pred_classes,
            'probabilities': pred_probs,
            'true_values': true_classes
        }
