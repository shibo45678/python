import pickle
import random
import warnings
from collections import defaultdict
import cloudpickle
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List
import re
import datetime
import numpy as np
from pydantic.v1 import validate_arguments
from pydantic import Field
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import pandas as pd
# from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


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
        self._temp_preprocessor = None  # 受保护（约定上外部不应直接访问)
        self.serialized_states = {}  # pipeline序列化状态

    def custom_inverse_transform(self, raw_predictions: Dict, task_config, use_saved, output_width, **kwargs):
        """
        智能逆转换：根据情况选择使用内存引用或保存的状态
        支持多预测任务（数值+分类）的逆转换

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

        logger.debug(f"任务预测: {len(raw_predictions)}个任务")

        processed_tasks = {}

        for task_name, task_pred in raw_predictions.items():
            if isinstance(task_pred, tf.Tensor):  # tf.Tensor / ndarray
                task_pred = task_pred.numpy()

            logger.debug(f"任务{task_name}原始形状: {task_pred.shape}")
            task_type = task_config.get(task_name).get('type', 'regression')

            """
            多步预测:
            1.回归 (batch,output_width,1) ；二分类(batch,output_width,1)->可压缩->2D np.squeeze
            2.多分类(batch,output_width,num_classes) -> 不压缩->保持3D（ argmax ->2D) 概率取最大的那个

            单步预测：
            1.回归（batch,1) ；二分类(batch,1) ->不压缩->2D
            2.多分类(batch,1,num_classes) -> 不压缩->保持3D（argmax->2D)

            逆标准化器：保证接受2D；
            逆编码器：保证接受3D，内部转换2D操作；
            """
            # 多步：（处理回归和二分类的压缩）目标2D输入 / 多分类保持3D输入
            if output_width > 1:
                if task_type in ['regression', 'binary_classification']:
                    if task_pred.shape[-1] == 1:  # 去除冗余的最后一个维度(samples, output_width,1)
                        task_pred = np.squeeze(task_pred, axis=-1)
                        logger.debug(f"regression任务{task_name}去除冗余后: {task_pred.shape}")
            # 单步：（不需处理回归和二分类，本身就是2D输入）/ 多分类3D输入

            if not use_saved and hasattr(self, '_temp_preprocessor'):
                task_result = self._inverse_transform_live(prediction=task_pred, target_column=task_name,
                                                           task_type=task_type, **kwargs)
            else:
                task_result = self._inverse_transform_from_saved(prediction=task_pred, target_column=task_name,
                                                                 task_type=task_type, **kwargs)

            logger.debug(f"任务{task_name}逆转换后: {task_result.shape}")
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
            scale_step_names = ['engineer_2', 'engineer_3']

        if transform_step_name is None:
            transform_step_name = 'engineer_1'

        result = prediction

        for step_name in scale_step_names:
            transformer = self._temp_preprocessor.pipelines_[pipeline_name].named_steps[step_name]

            # 逆标准化
            if step_name == 'engineer_2':
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
            elif step_name == 'engineer_3':
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
        # 逆标准化 / 逆编码
        if scale_step_names is None:
            scale_step_names = ['engineer_2', 'engineer_3']
        # 逆转换
        if transform_step_name is None:
            transform_step_name = ['engineer_1']

        result = prediction

        for step_name in scale_step_names:
            if pipeline_name in self.serialized_states and step_name in self.serialized_states[pipeline_name]:
                state_info = self.serialized_states[pipeline_name][step_name]

                # 从pickle重建transformer
                if state_info.get('pickled'):
                    transformer = pickle.loads(state_info['pickled'])

                    if hasattr(transformer, 'custom_inverse_transform'):
                        if step_name == 'engineer_2':
                            valid_col = transformer.without_outlier_missing_columns_
                            if target_column is not None and task_type == 'regression' and target_column in valid_col:
                                result = transformer.custom_inverse_transform(scaled_data=result,
                                                                              target_column=target_column)

                            elif target_column is not None and task_type == 'binary_classification' and target_column not in valid_col:
                                threshold = 0.5
                                result = (result > threshold).astype(int)

                            else:
                                logger.debug(f"目标列{target_column}不需要数值列的逆标准化转换或者二分阈值管理")
                        elif step_name == 'engineer_3':
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

        logger.debug(f"_inverse_transform_saved 结束，返回类型: {type(result)}")
        return result

    def add_timestamps(self, predictions: Dict, historical_timestamps, input_width: int, output_width: int, shift: int,
                       freq: str, save_path, mode='train'):
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

        if mode == 'train':
            # 训练窗口数（有真实标签的） len -total_window + 1 =1548 - 34 +1 =1515
            n_windows = len(historical_timestamps) - (input_width + shift + output_width - 1) + 1
        else:
            # 预测样本数（只需要输入，没有滑动）
            n_windows = len(historical_timestamps) - (input_width) + 1

        window_start_times = []  # 每个窗口的基准时间(input_width的最后一个点）
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

        return self._create_result_df(predictions, window_start_times, future_timestamps, save_path)

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

    def _create_result_df(self, predictions, window_start_times: list, future_timestamps: list, save_path: str):
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
                    'input_end': start_times,
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

        results_df.to_csv(save_path, index=False)

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
        combined.to_csv(f"/Users/shibo/AL/NeuralNetwork/temperature_forecasting/data/final/true_pred.csv")

        # 逐时间步
        step_res = MetricsCalculator.calc_every_pair(data=combined, task_names=task_names)

        return step_res

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
            save_dir = self.config.get('save_dir', '/Users/shibo/AL/NeuralNetwork/saved_model_state')
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
        state_file = Path(save_dir) / 'pipeline_states.cpkl'

        save_data = {
            'serialized_states': self.serialized_states,
            'config': self.config,
            'saved_at': datetime.datetime.now().isoformat()
        }
        with state_file.open('wb') as f:
            cloudpickle.dump(save_data, f)
        logger.info(f"Pipeline状态已保存到: {state_file}")
        return save_data


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
                            time_column: str = None):
        """""""""
         mape 先处理统计对，再计算
        ---------------------------

        参数：
        predictions ： 需要经过逆转换处理 {task_name: predictions}的格式 Mape / 或者未逆标准化的MAE MSE计算。每个任务有滑动窗口按需处理
        actual_data：原始的数据取对应的时间列 + 任务列，
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
                        if isinstance(pred_value, np.ndarray):
                            if pred_value.ndim != 1:
                                raise ValueError(f"期望1D数组，但shape{pred_value.shape}")
                            pred = np.float64(pred_value.item())
                        else:
                            pred = np.float64(pred_value)

                        predictions_by_time[time_point].append(pred)

            result[tk] = predictions_by_time

        return result

    @staticmethod
    def smape_calculator(predictions: Dict, actual_data: pd.DataFrame, input_width: int, shift: int,
                         time_column: str = None,
                         level: str = 'o'):
        """""""""      
        A.业务指标MAPE：
            逆标准化后的 predictions（predictions_by_time 字典）
            单时间点 mape  + 日/月/年级别 mape + 整体 mape
            1. 单时间点：某时间点在预测中出现多次，先求预测的均值，再和actual_data的时间点进行计算；
            2. 日级别：聚合某日内所有预测时间点，再整理actual_data的日级别真实值，求ape，MAPE为同级别所有ape的均值mean(ape);
            3. 整体：基础对
        level: ‘D’ 代表 日级别 ,‘O' 单时间点级别，‘M’月级别，'Y'年级别，'A'整体
        """

        predictions_by_time = MetricsCalculator.predictions_by_time(predictions=predictions, actual_data=actual_data,
                                                                    input_width=input_width,
                                                                    shift=shift, time_column=time_column)
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
        A.还原业务mae level='operational'
        B.技术指标 MSE MAE：
          仍然是[标准化]的 predictions_by_time 字典
          最基本的计算单元是每个预测值与其对应实际值的比较(标准化状态下的）。
          无论什么层级，最终都是这些基础对的统计。
        """

        predictions_by_time = MetricsCalculator.predictions_by_time(predictions=predictions, actual_data=actual_data,
                                                                    input_width=input_width,
                                                                    shift=shift, time_column=time_column)

        result_mae_mse = {}
        result_mae = {}

        if level == 'operational':
            for tk, pred in predictions_by_time.items():  # pairs 是时间点明细
                MAE, MSE, RMSE = MetricsCalculator._calc_hierarchical_mae_mse(pred, actual_data, tk,
                                                                              level='operational')
                logger.info(f"业务解释 任务{tk}(整体数据) --  mae_original：{MAE:.6f} (基础对）")
                result_mae[tk] = MAE
            return result_mae

        else:
            for tk, pred in predictions_by_time.items():
                MAE, MSE, RMSE, pairs, mae_mse_ = MetricsCalculator._calc_hierarchical_mae_mse(pred, actual_data, tk,
                                                                                               level='o')
                logger.info(f'模型拟合指标 任务{tk}(整体数据) --  MAE：{MAE:.6f},MSE:{MSE:.6f},RMSE:{RMSE:.6f}（基础对）')

                result_mae_mse[tk] = mae_mse_
                result_mae[tk] = MAE

            return result_mae_mse, result_mae

    @staticmethod
    def download_data(result_mae_mse: Dict, result_mape: Dict, level_mae_mse: str, level_mape: str):
        """1个任务一个工作簿，1个指标占1个sheet"""
        if level_mae_mse.lower() != level_mape.lower():
            logger.warning(f"mae_mse 计算指标维度不统一")

        for tk, result in result_mae_mse.items():
            file = f'/Users/shibo/AL/NeuralNetwork/temperature_forecasting/data/intermediate/analyze/{tk}_metrics.xlsx'

            if tk in result_mape.keys():
                mae_mse = result
                mape_ = result_mape.get(tk, None)
                mape = mape_.get(f'details_{level_mape}', pd.DataFrame)
                mape_analyze = mape_.get(f'mape_analyze_{level_mape}', {})

                mape_analyze_consecutive = mape_analyze.get('consecutive', [])
                mape_analyze_consecutive_df = pd.DataFrame(mape_analyze_consecutive)

                mape_weakness = mape_analyze.get('weaknesses', {}).get('spike_points_mape', [])
                mape_weakness_df = pd.DataFrame(mape_weakness, columns=['spike_points_mape'])

                mape_pattern = mape_analyze.get('patterns', {})

                mape_pattern_high_error_hours = mape_pattern.get('high_error_hours', {})
                mape_pattern_high_error_hours = pd.DataFrame({'high_error_hour': mape_pattern_high_error_hours.keys(),
                                                              'counts': mape_pattern_high_error_hours.values()})

                mape_pattern_high_error_months = mape_pattern.get('high_error_month', {})
                mape_pattern_high_error_months = pd.DataFrame(
                    {'high_error_month': mape_pattern_high_error_months.keys(),
                     'counts': mape_pattern_high_error_months.values()})

                mape_pattern_high_error_years = mape_pattern.get('high_error_year', {})
                mape_pattern_high_error_years = pd.DataFrame(
                    {'mape_pattern_high_error_year': mape_pattern_high_error_years.keys(),
                     'counts': mape_pattern_high_error_years.values()
                     })

                mape_pattern_actual_range = mape_pattern.get('actual_range', {})
                mape_pattern_actural_range = pd.DataFrame({'actural_info': mape_pattern_actual_range.keys(),
                                                           'values': mape_pattern_actual_range.values()})

                mape_pattern_high_error_count = mape_pattern.get('count', 0)
                mape_pattern_high_error_rate = mape_pattern.get('error_rate', 0)
                logger.debug(
                    f"{tk}_smape计算出的>50的高异常值，数量{mape_pattern_high_error_count}，占比：{mape_pattern_high_error_rate}")

                sheet_data = {
                    'mape': mape,
                    'mae_mse': mae_mse,
                    'mape_consecutive': mape_analyze_consecutive_df,
                    'mape_weakness': mape_weakness_df,
                    'mape_pattern_high_error_hours': mape_pattern_high_error_hours,
                    'mape_pattern_high_error_months': mape_pattern_high_error_months,
                    'mape_pattern_high_error_years': mape_pattern_high_error_years,
                    'mape_pattern_actural_range': mape_pattern_actural_range,
                }

                with pd.ExcelWriter(file, engine='openpyxl') as writer:
                    for sheet_name, df in sheet_data.items():
                        df.to_excel(writer, sheet_name=sheet_name)

    @staticmethod
    def _calc_hierarchical_mape(predictions_by_time: Dict, actual_data: pd.DataFrame, level: str, tk: str,
                                time_column: str):
        # 整个数据集mape
        MAPE = MetricsCalculator._calc_dataset_mape(pre_dict=predictions_by_time, actual_data=actual_data, tk=tk,
                                                    time_col=time_column)
        logger.info(f"业务解释 任务{tk}(整体数据) -- smape: {MAPE:.6f} (基础对ape均值）")

        # 3.1 mape: add 时间点维度
        if level.lower() == 'o':
            avg_timepoint_predictions = []
            for time_point, preds in predictions_by_time.items():
                avg_timepoint_predictions.append(
                    {'timestamp': time_point,
                     f'{tk}_pred': np.mean(preds)
                     }
                )
                # pred = np.array(preds, dtype='float64')
                # if np.any(pred > 1e+3):
                #     logger.warning(f"{time_point}_pred中存在异常{pred}")

            avg_predictions_df = pd.DataFrame(avg_timepoint_predictions).set_index('timestamp')

            timepoint_actual_data = actual_data.set_index(time_column)

            timepoint_mape, timepoint_details = MetricsCalculator.calc_mape(avg_predictions_df,
                                                                            timepoint_actual_data[tk])
            logger.info(f"业务解释 任务{tk}(原始粒度) -- smape_o:{timepoint_mape:.6f}(时间点ape均值）")
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

            avg_predictions_df = pd.DataFrame(avg_predictions)  # 列表里面是字典，直接pd.DataFrame ,如果列表里面是df就要pd.concat()
            level_prediction_data = avg_predictions_df.groupby('timestamp').mean(numeric_only=True)

            level_mape, level_details = MetricsCalculator.calc_mape(level_prediction_data, level_actual_data[tk])
            daily_analyze = MetricsCalculator.analyze_mape(details=level_details)

            logger.info(f"业务解释 任务{tk}({level}级别) -- smape_{level}:{level_mape:.6f}(时间点ape均值）")
            result = {
                f'details_{level}': level_details,
                f'mape_analyze_{level}':
                    {'patterns': daily_analyze[0],
                     'consecutive': daily_analyze[1],
                     'weak nesses': daily_analyze[2]}
            }

        return result

    @staticmethod
    def _calc_dataset_mape(pre_dict, actual_data, tk, time_col):
        actual_data = actual_data.set_index(time_col)
        actual_data = actual_data[tk]

        pairs_res = []
        for i, (time_point, pred_list) in enumerate(pre_dict.items()):
            actual = actual_data.loc[time_point]

            # pred = np.array(pred_list, dtype='float64')
            # if np.any(pred > 1e+3):
            #     logger.warning(f"{time_point}_pred中存在异常{pred}")

            for pred in pred_list:
                denominator = (abs(actual) + abs(pred)) / 2
                eps = 1e-8
                if denominator < eps:
                    ape = 0
                else:
                    ape = abs(pred - actual) / denominator * 100
                pairs_res.append(ape)

                if pred > 1e+8:
                    print(f"time{time_point}:pred{pred},actual{actual}")

        MAPE = np.mean(pairs_res)
        return MAPE

    @staticmethod
    def _calc_hierarchical_mae_mse(predictions_by_time: Dict, actual_data, tk: str, level: str):
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

        # 整体数据集
        MAE = pairs_df['abs_error'].mean()
        MSE = pairs_df['squared_error'].mean()
        RMSE = np.sqrt(MSE)

        if level.lower() == 'operational':  # 进业务计算，后续不用计算
            return MAE, MSE, RMSE

        else:
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
                                                                 rmse=('squared_error', lambda x: np.sqrt(x.mean())
                                                                       # rmse
                                                                       ))
            return MAE, MSE, RMSE, pairs_df, hierarchical_metrics

    @staticmethod
    def calc_mape(handled_pred: pd.DataFrame, handled_actual: pd.Series):
        """
        参数：
        handled_pred： 转换的时间维度 的预测值的均值（已处理）
        handled_actual：转换的时间维度 的真实值（已处理），actual 包含1列tk(任务列），索引是level
        接收单任务
        """

        detailed_results = pd.DataFrame()

        avg_pred = handled_pred.iloc[:, 0].values

        # 截掉开头非预测时间点
        actual = handled_actual.values[-len(avg_pred):]

        detailed_results['timestamp'] = handled_pred.index
        detailed_results['actual'] = actual
        detailed_results['avg_pred'] = avg_pred

        mask = ((~np.isnan(actual) & ~np.isnan(avg_pred)) &
                (~np.isinf(actual) & ~np.isinf(avg_pred)) &
                (actual != 0))

        detailed_results['level_ape'] = np.nan

        ape_values = np.abs(actual[mask] - avg_pred[mask]) / ((np.abs(actual[mask]) + np.abs(avg_pred[mask])) / 2) * 100
        detailed_results.loc[mask, 'level_ape'] = ape_values

        # 计算有效ape的平均，不能先求和再计算
        if len(detailed_results) > 0:
            mape = np.mean(ape_values)  # 该级别
        else:
            mape = float('nan')

        return mape, detailed_results

    @staticmethod
    def analyze_mape(details: pd.DataFrame):
        """每次接收单个任务"""

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

                    if end - start + 1 >= 3:  # 至少连续3个错误点（持续2小时）
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
