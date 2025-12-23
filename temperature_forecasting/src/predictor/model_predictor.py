import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=全部显示, 1=隐藏INFO, 2=隐藏WARNING, 3=隐藏ERROR
import tensorflow as tf

class TrainedModelPredictor:
    """专门用于预测的类，不依赖训练状态"""

    def __init__(self,checkpoint_path):
        self.model = None
        self.checkpoint_path = checkpoint_path

        if checkpoint_path:
            self.load_model()

    def load_model(self, checkpoint_path=None):
        """加载模型"""

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        if checkpoint_path is None:
            raise ValueError("未指定检查点路径")

        # 查找并加载模型文件
        if os.path.isdir(checkpoint_path):
            # 在目录中查找模型文件
            for file in os.listdir(checkpoint_path):
                if file.endswith(('.keras', '.h5')):
                    model_path = os.path.join(checkpoint_path, file)
                    self.model = tf.keras.models.load_model(model_path)
                    self.checkpoint_path = model_path
                    return self.model
            raise FileNotFoundError(f"在目录中找不到模型文件: {checkpoint_path}")
        else:
            # 直接加载文件
            self.model = tf.keras.models.load_model(checkpoint_path)
            return self.model

    def predict(self, new_data):
        """进行预测"""
        if self.model is None:
            self.load_model()
        return self.model.predict(new_data)


