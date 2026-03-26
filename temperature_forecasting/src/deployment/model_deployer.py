from collections import defaultdict
from datetime import datetime
import os, cloudpickle, shutil
from pathlib import Path
import logging
import re
import pandas as pd
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class DeploymentManager:
    def __init__(self, model_name, bestpoint, preprocessor, postprocessor, model_config, window_config,
                 window_generator, mae_dict):
        self.model_name = model_name
        self.best_checkpoint = bestpoint
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model_config = model_config
        self.window_config = window_config # 含mode
        self.window_generator = window_generator # forecast
        self.mae_dict = mae_dict  # 字典格式 task:测试集MAE

    def save(self, deployment_path):

        # 创建部署目录
        self.deploy_path = Path(deployment_path)
        self.deploy_path.mkdir(parents=True, exist_ok=True)  # 父目录不存在自动创建

        # if not self.deploy_path.exists():
        #     logger.warning(f'检查到部署的SavedModel文件不存在{self.deploy_path}')

        # 1. 保存模型（从检查点复制 SavedModel）
        match = int(re.search(r'tf_checkpoints_stage(\d+)', self.best_checkpoint).group(1))
        source_savedmodel = Path(self.best_checkpoint) / f'saved_model_stage{match}'
        target_savedmodel = self.deploy_path / f'{self.model_name}_saved_model_stage{match}'

        if source_savedmodel.exists():
            if target_savedmodel.exists():
                shutil.rmtree(target_savedmodel)
            shutil.copytree(source_savedmodel, target_savedmodel)
            logger.info(f"已复制 SavedModel: {target_savedmodel}")
        # else:
        #     if not hasattr(self, '_prediction_model'):
        #         self._prediction_model = self.load_best_model()
        #     self._prediction_model.save(str(target_savedmodel), save_format='tf')
        #     logger.info(f"已创建新的 SavedModel: {target_savedmodel}")


        # 2. 保存CompletePreprocessor
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            preprocessor_path = os.path.join(deployment_path, f'{self.model_name}_preprocessor.cpkl')
            with open(preprocessor_path, 'wb') as f:
                cloudpickle.dump(self.preprocessor, f)
            logger.info(f"已保存 CompletePreprocessor:{preprocessor_path}")

        # 3. 保存TimeSeriesPostProcessor
        if hasattr(self, 'postprocessor') and self.postprocessor is not None:
            postprocessor_path = os.path.join(deployment_path, f'{self.model_name}_postprocessor.cpkl')
            with open(postprocessor_path, 'wb') as f:
                cloudpickle.dump(self.postprocessor, f)

            # 3.1 保存postprocessor的serialized_states
            if hasattr(self.postprocessor, 'serialized_states'):
                states_path = os.path.join(deployment_path, f'{self.model_name}_pipeline_states.cpkl')
                with open(states_path, 'wb') as f:
                    cloudpickle.dump(self.postprocessor.serialized_states, f)

        # 4. 保存配置信息
        config = {
            'model_config': self.model_config,
            'window_config': self.window_config,  # 预测窗口生成器状态
            'preprocessor_info': {
                # 1. 基本信息
                'type': type(self.preprocessor).__name__ if hasattr(self, 'preprocessor') else None,
                'serialization_method': 'cloudpickle',
                'file_path': 'preprocessor.cpkl',

                # 2.预处理步骤
                'processing_steps': self._get_processing_steps(self.mae_dict),

                # 3.特征信息（用于API 验证）
                'feature_info': self._get_features_info()
            },
            'deployment_info':
                {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'framework': 'tensorflow',
                    'requires_preprocessing': True
                }
        }

        config_path = os.path.join(deployment_path, f'{self.model_name}_deployment_config.cpkl')
        with open(config_path, 'wb') as f:
            cloudpickle.dump(config, f)

        # 5.保存窗口生成器
        if hasattr(self, 'window_generator') and self.window_generator is not None:
            window_gen_path = os.path.join(deployment_path, f'{self.model_name}_window_generator.cpkl')
            with open(window_gen_path, 'wb') as f:
                cloudpickle.dump(self.window_generator, f)
            logger.info(f"已保存 WindowGenerator:{window_gen_path}")

        # 6. 保存requirements.txt（依赖信息）
        self._save_requirements(deployment_path)

        logger.info(f"部署包已保存到：{deployment_path}")
        return deployment_path

    def _get_processing_steps(self, mae_dict):
        steps_info = {}
        if not hasattr(self, 'preprocessor') or self.preprocessor is None:
            return steps_info

        try:
            preprocessor = self.preprocessor

            if hasattr(preprocessor, 'pipelines_') and preprocessor.pipelines_:

                # 1. 检查每个步骤是否拟合
                for pipeline_name, pipeline in preprocessor.pipelines_.items():
                    pipeline_info = {
                        'pipeline_type': pipeline.__class__.__name__,
                        'steps': [],
                        'is_fitted': check_is_fitted(pipeline)  # 静默返回None 未拟合返回NotFittedError
                    }

                    if hasattr(pipeline, 'steps'):
                        for step_name, transformer in pipeline.steps:
                            step_info = {
                                'step_name': step_name,
                                'transformer_type': transformer.__class__.__name__,
                                'is_fitted': check_is_fitted(transformer),
                                'statistices': self._get_transformer_statistics(transformer, mae_dict)
                            }
                            pipeline_info['steps'].append(step_info)

                    steps_info[pipeline_name] = pipeline_info

            if hasattr(preprocessor, 'cleaner_') and preprocessor.cleaner_:
                for cleaner_name, cleaners in preprocessor.cleaner_.items():

                    cleaner_info = {
                        'cleaner_name':cleaner_name,
                        'steps': [],
                    }
                    for i,cl in enumerate(cleaners):
                        instance = cl.get(f'step_{i}')
                        cl_info = {
                            'transformer_type':instance.__class__.__name__,
                            'step_name': f'step_{i}',
                            'is_processed':cl.get('is_processed_')
                        }
                        cleaner_info['steps'].append(cl_info)

                    steps_info[cleaner_name] = cleaner_info

            return steps_info

        except Exception as e:
            logger.warning(f"获取预处理步骤信息时出错: {e}")
            return {'error': str(e)}

    def _get_features_info(self):
        time_col = self.model_config.get('time_column')

        if hasattr(self.preprocessor, 'get_specific_attribute'):
            num_cols_input = self.preprocessor.get_specific_attribute(6, 'engineer_2',
                                                                      'numeric_columns_')  # 取第5个class的第4步的属性
            num_cols_inverse = self.preprocessor.get_specific_attribute(6, 'engineer_2',
                                                                        'without_outlier_missing_columns_')
            cat_cols_input = self.preprocessor.get_specific_attribute(6, 'engineer_3', 'categorical_columns_')

            columns = {
                'num_cols_input': num_cols_input,
                'cat_cols_input': cat_cols_input,
                'num_cols_inverse': num_cols_inverse,
                'time_col': time_col
            }

            return columns

    def _get_transformer_statistics(self, transformer, mae_dict):
        statistics = defaultdict(list)
        trans_type = transformer.__class__.__name__

        if trans_type == 'UnifiedFeatureScaler':
            scaling_config = getattr(transformer, 'scaling_config_')

            for col, config in scaling_config.items():

                method = config.get('method')
                stats = config.get('stats')
                scaler = config.get('scaler')

                statistics[col].append({'method':method})

                if method == 'standard':  # 标准 标准化 从scaler里面获取统计量即可
                    if hasattr(scaler, 'mean_'):
                        statistics[col].append({
                            'mean': scaler.mean_.tolist()
                        })
                    if hasattr(scaler, 'var_'):
                        statistics[col].append(
                            {'var': scaler.var_.tolist()})

                    if hasattr(scaler, 'scale_'):  # 标准差
                        statistics[col].append({
                            'std': scaler.scale_.tolist()
                        })
                        if col in mae_dict.keys():
                            mae_scaled = mae_dict.get(col)[0]
                            logger.info(f"label {col}列:"
                                        f"MAE_original = MAE_standard {mae_scaled:.5f} * train_std {scaler.scale_.tolist()[0]:.5f}"
                                        f" = {mae_scaled * scaler.scale_.tolist()[0]:.5f}")

                elif method == 'minmax':
                    if hasattr(scaler, 'data_min_'):
                        statistics[col].append({
                            'data_min': scaler.data_min_.tolist()
                        })
                    if hasattr(scaler, 'data_max_'):
                        statistics[col].append({
                            'data_max_': scaler.data_max_.tolist()
                        })

                    if hasattr(scaler, 'data_range_'):
                        statistics[col].append(
                            {'data_range': scaler.data_range_.tolist()})
                        if col in mae_dict.keys():
                            mae_scaled = mae_dict.get(col)[0]
                            if hasattr(scaler,'feature_range'):
                                a=scaler.feature_range[0]
                                b=scaler.feature_range[1]
                                logger.info(f"label {col}列:"
                                            f"MAE_original = MAE_minmax {mae_scaled:.5f} * train_minmax {scaler.data_range_.tolist()[0]:.5f} / (b-a) ({b}-{a})"
                                            f" = {mae_scaled * scaler.scale_.tolist()[0] / (b-a) :.5f}")

                elif method == 'robust':
                    if hasattr(scaler, 'center_'):  # median
                        statistics[col].append({'center': scaler.center_.tolist()})  # 中位数

                    if hasattr(scaler, 'scale_'):  # iqr
                        statistics[col].append({'scale': scaler.scale_.tolist()}) # Iqr

                        if col in mae_dict.keys():
                            mae_scaled = mae_dict.get(col)[0]
                            logger.info(f"label {col}列:"
                                        f"MAE_original = MAE_robust {mae_scaled:.5f} * train_iqr {scaler.scale_.tolist()[0]:.5f}"
                                        f" = {mae_scaled * scaler.scale_.tolist()[0]:.5f}")


                elif method == 'manual_robust':  # 非标 标准化  <=4 分母的iqr替换成std
                    if hasattr(config, 'median'):
                        statistics[col].append({
                            'median': stats.median.tolist()
                        })

                    if hasattr(config, 'std'):
                        statistics[col].append({
                            'std': stats.std.tolist()}
                        )
                        if col in mae_dict.keys():
                            mae_scaled = mae_dict.get(col)[0]
                            logger.info(
                                f"label {col}列:"
                                f"MAE_original = MAE_manual_robust {mae_scaled:.5f} * train_std {stats.std.tolist()[0]:.5f}"
                                f" = {mae_scaled * stats.std.tolist()[0]:.5f}")



            scaler_info = pd.DataFrame(
                {'column': statistics.keys(),
                 'info': statistics.values()})
            scaler_info.to_csv(
                '/Users/shibo/AL/NeuralNetwork/temperature_forecasting/data/intermediate/scaler_info.csv')

        if trans_type == 'CategoricalEncoding':
            encoders = getattr(transformer, 'encoders_')
            for col, encoder in encoders.items():
                if hasattr(encoder, 'classes_'):
                    statistics[col].append({
                        'type': 'LabelEncoder',
                        'n_classes': len(encoder.classes_),
                        'classes': encoder.classes_.tolist(),
                        'mapping': {cls: i for i, cls in enumerate(encoder.classes_)}
                    })

        return statistics if statistics else None

    def _save_requirements(self, deploy_path):
        requirements_path = os.path.join(deploy_path, f'{self.model_name}_requirements.txt')

        requirements = [
            '# ==================',
            '# 神经网络时间序列预测 - 生产环境依赖',
            '# ===================',
            '',
            '# ==深度学习框架================ ',
            'tensorflow==2.20.0',
            'keras==3.13.0'
            '',
            '# ==数据处理=================== ',
            'pandas==2.3.3',
            'numpy==2.4.0',
            'scipy==1.16.3',
            '',
            '# ==机器学习工具================ ',
            'scikit-learn==1.8.0',
            'joblib==1.5.3',
            '',
            '# ==序列化与部署================ ',
            'cloudpickle==3.1.2',
            'protobuf==6.33.2',
            '',
            '# ==TensorFlow支持库===================== ',
            'absl-py==2.3.1',
            'grpcio==1.76.0',
            'gast==0.7.0',
            '',
            '# ==工具库===================== ',
            'python-dateutil==2.9.0.post0',
            'pytz==2025.2',
            'matplotlib==3.10.8',
            'plotly==6.5.0',
            'pydantic==2.12.5',
            'pydantic_core==2.41.5',
            'pytest==9.0.2',
            'openpyxl==3.1.5',
            '',
            '# ==环境说明===================== ',
            '# Python 版本：3.11',
            '# 操作系统：',
            '# 安装命令：pip install -r requirements.txt',
            '# 验证命令：python -c "import tensorflow as tf:print(tf.__version__)"'

        ]
        try:
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(requirements))
            logger.info(f'已生成生产环境 requirements.txt')
            logger.info(f'包含 {[l for l in requirements if l.strip() and not l.startswith("#")]}核心包')
        except Exception as e:
            logger.error(f'保存requirements.txt失败：{e}')
