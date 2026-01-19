import json
import os
import cloudpickle
import logging

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=全部显示, 1=隐藏INFO, 2=隐藏WARNING, 3=隐藏ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关键：禁用 Keras 3 的自动检测
import tensorflow as tf


class TrainedModelPredictor:
    """专门用于预测的类，不依赖训练状态 """

    def __init__(self, deployment_package_path: str):
        self.deployment_path = deployment_package_path
        self.model = None
        self.preprocessor = None
        self.postprocessor = None
        self.window_generator = None
        self.config = None

    def load(self):
        """加载所有组件 使用 TensorFlow SavedModel"""
        # 1. 加载配置
        config_path = os.path.join(self.deployment_path, 'deployment_config.cpkl')
        with open(config_path, 'rb') as f:
            self.config = cloudpickle.load(f)

        # 2. 加载 预处理器
        preprocessor_path = os.path.join(self.deployment_path, 'preprocessor.cpkl')
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = cloudpickle.load(f)

        # 3. 加载模型
        model_path = os.path.join(self.deployment_path, 'saved_model')
        self.model = tf.saved_model.load(model_path)
        logger.info(f"已加载 TensorFlow SavedModel")

        if hasattr(self.model, 'signatures'):
            self.serving_fn = self.model.signatures['serving_default']
        else:
            self.serving_fn = self.model

        # 4. 加载窗口生成器
        window_gen_path = os.path.join(self.deployment_path, 'window_generator.cpkl')
        if os.path.exists(window_gen_path):
            with open(window_gen_path, 'rb') as f:
                self.window_generator = cloudpickle.load(f)
                logger.info("已加载窗口生成器")
        else:
            raise FileNotFoundError("缺少 window_generator.cpkl")

        # 5. 加载后处理器
        postprocessor_path = os.path.join(self.deployment_path, 'postprocessor.cpkl')
        if os.path.exists(postprocessor_path):
            with open(postprocessor_path, 'rb') as f:
                self.postprocessor = cloudpickle.load(f)

        return self

    def forecast(self, new_features, labels=None):

        processed_data, _ = self.preprocessor.transform_predict(features=new_features, labels=None)

        # 处理时间列
        datetime_cols = processed_data.select_dtypes(include=['datetime64']).columns
        input_cols_ = [col for col in list(processed_data.columns) if col not in datetime_cols]
        processed_data_ = processed_data[input_cols_]

        window_data = self.window_generator.createDataset(processed_data_)  # 全部数值列

        # 使用TensorFlow SavedModel预测 ，而不是predict
        # SavedModel 的签名期望的是 Tensor，不是 Dataset，所以需要从 Dataset 中提取 Tensor
        # Keras 的predict可以接受Dataset
        numeric_input = None

        for batch in window_data.take(1):

            if isinstance(batch, tuple):
                numeric_input = batch[0]
            else:
                numeric_input = batch
            break

        print(f"提取的 numeric_input 形状: {numeric_input.shape}")
        raw_outputs = self.serving_fn(numeric_input=numeric_input,categorical_segments_input=categorical_input)

        return self._process_multi_output(raw_outputs)

    def _process_multi_output(self, raw_outputs):

        task_config = self.config.get('model_config').get('output_config')
        output_width = self.config.get('model_config').get('output_width')
        pipeline_name = 'pipeline_4'
        step_names = ['engineer_3', 'engineer_4']

        # SavedModel 通常返回字典，如 {'output_0': tensor, 'output_1': tensor}
        numpy_pred = None
        if isinstance(raw_outputs, dict):
            if self.postprocessor:
                numpy_pred = self.postprocessor.custom_inverse_transform(raw_predictions=raw_outputs, use_saved=False,
                                                                         task_config=task_config,
                                                                         output_width=output_width,
                                                                         pipeline_name=pipeline_name,
                                                                         step_names=step_names)

        return numpy_pred
