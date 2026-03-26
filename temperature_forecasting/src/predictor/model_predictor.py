import os
import cloudpickle
import logging

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=全部显示, 1=隐藏INFO, 2=隐藏WARNING, 3=隐藏ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关键：禁用 Keras 3 的自动检测
import tensorflow as tf


class TrainedModelPredictor:
    """专门用于预测的类，不依赖训练状态 """

    def __init__(self, deployment_package_path: str,model_name:str,stage:int):
        self.deployment_path = deployment_package_path
        self.model_name=model_name
        self.stage=stage

        self.model = None
        self.preprocessor = None
        self.postprocessor = None
        self.window_generator = None
        self.config = None
        self.processed_data=None


    def load(self):
        """加载所有组件 使用 TensorFlow SavedModel"""

        # 1. 加载配置
        config_path = os.path.join(self.deployment_path, f'{self.model_name}_deployment_config.cpkl')
        with open(config_path, 'rb') as f:
            self.config = cloudpickle.load(f)

        # 2. 加载 预处理器
        preprocessor_path = os.path.join(self.deployment_path, f'{self.model_name}_preprocessor.cpkl')
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = cloudpickle.load(f)

        # 3. 加载模型
        model_path = os.path.join(self.deployment_path, f'{self.model_name}_saved_model_stage{self.stage}')

        self.model = tf.saved_model.load(model_path)
        # print(self.model.signatures['serving_default'].inputs)
        # print(self.model.signatures['serving_default'].outputs)

        logger.info(f"已加载 TensorFlow SavedModel")

        if hasattr(self.model, 'signatures'):
            self.serving_fn = self.model.signatures['serving_default']
        else:
            self.serving_fn = self.model

        # 4. 加载窗口生成器
        window_gen_path = os.path.join(self.deployment_path, f'{self.model_name}_window_generator.cpkl')
        if os.path.exists(window_gen_path):
            with open(window_gen_path, 'rb') as f:
                self.window_generator = cloudpickle.load(f)
                logger.info("已加载窗口生成器")
        else:
            raise FileNotFoundError("缺少 window_generator.cpkl")

        # 5. 加载后处理器
        postprocessor_path = os.path.join(self.deployment_path, f'{self.model_name}_postprocessor.cpkl')
        if os.path.exists(postprocessor_path):
            with open(postprocessor_path, 'rb') as f:
                self.postprocessor = cloudpickle.load(f)

        return self


    def forecast(self, new_features, labels=None):

        self.processed_data, _ = self.preprocessor.transform_predict(features=new_features, labels=labels)

        # 处理时间列
        datetime_cols = self.processed_data.select_dtypes(include=['datetime64']).columns
        input_cols_ = [col for col in list(self.processed_data.columns) if col not in datetime_cols]
        processed_data_ = self.processed_data[input_cols_]

        window_data = self.window_generator.createDataset(processed_data_)  # 最新一批的Dataset  (num+cat)

        # 使用TensorFlow SavedModel预测 ，而不是predict(Keras 的predict可以接受Dataset)
        # SavedModel 的签名期望的是 Tensor，不是 Dataset，
        # 所以需要从 Dataset 中提取 Tensor

        cat_cols = self.config.get('model_config',{}).get('categorical_columns',[])

        for batch in window_data.take(1):
            if isinstance(batch, tuple): # <tf.Tensor: shape=(1, 6), dtype=int32, numpy=array([[3, 4, 4, 5, 4, 4]], dtype=int32)>
                params_tensor = {'numeric_input': batch[0]}

                if cat_cols:
                    cat_params = [f'categorical_{cat}_input' for cat in cat_cols]
                    cat_tensor = list(batch[1:])
                    categorical_input = [{p:t} for p,t in zip(cat_params,cat_tensor)]

                    for cat_input in categorical_input:
                        params_tensor.update(cat_input)

                    raw_outputs = self.serving_fn(**params_tensor)
                else:
                    raw_outputs = self.serving_fn(**params_tensor)
            else:
                numeric_tensor = batch
                raw_outputs = self.serving_fn(numeric_input=numeric_tensor)
            break

        return self._process_multi_output(raw_outputs)



    def _process_multi_output(self, raw_outputs):

        task_config = self.config.get('model_config').get('output_config')
        output_width = self.config.get('model_config').get('output_width')
        input_width = self.config.get('model_config').get('input_width')
        shift = self.config.get('model_config').get('shift')
        time_column =self.config.get('model_config').get('time_column')

        pipeline_name = 'pipeline_6'
        step_scale = ['engineer_2', 'engineer_3']
        step_trans ='engineer_1'

        # SavedModel 通常返回字典，如 {'output_0': tensor, 'output_1': tensor}
        if not isinstance(raw_outputs, dict):
            logger.warning(f"输出格式非字典格式，检查")

        if self.postprocessor:
            inversed_pred = self.postprocessor.custom_inverse_transform(raw_predictions=raw_outputs,
                                                                     use_saved=False,
                                                                     task_config=task_config,
                                                                     output_width=output_width,
                                                                     pipeline_name=pipeline_name,
                                                                     scale_step_names=step_scale,
                                                                     transform_step_name =step_trans)

            save_path = os.path.join(self.deployment_path,'deploy_predictions.csv')
            final_pred_results, predictions_dict = self.postprocessor.add_timestamps(
                mode ='forecast',
                predictions=inversed_pred,
                historical_timestamps=self.processed_data[time_column],
                input_width=input_width,
                output_width=output_width,
                freq='h',
                shift=shift,
                save_path = save_path
            )


            return final_pred_results,save_path


