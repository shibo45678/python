import os
import numpy as np
def ensure_tf_settings():
    """确保环境变量被设置（可在任何地方调用）"""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('PYTHONHASHSEED', '42')
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
ensure_tf_settings()
import tensorflow as tf

tf.config.experimental.enable_op_determinism() # 启用确定性
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42) # 最后设置种子（作为fallback）



class TensorFlowConfig:
    """TensorFlow配置类"""

    @staticmethod
    def setup_environment():
        """设置TensorFlow运行环境"""
        # 配置GPU内存增长（如果有GPU）
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    @staticmethod
    def check_performance():
        print("TensorFlow版本:", tf.__version__)
        print("可用GPU数量:", len(tf.config.experimental.list_physical_devices('GPU')))






