import os
import random
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field, PositiveInt, field_validator
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def set_global_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


class MethodConfig(BaseModel):
    units: List[PositiveInt] = Field(default=[192, ], description="滤波器数量，len控制lstm的层数")
    return_sequences: List[bool] = Field(default=[False, ],
                                         description="是否只在最后一个时间步产生输出，对应LSTM层数")
    input_width: int = Field(default=6)
    output_width: int = Field(default=5, description="输出时间步步长。例如 5 表示预测5个时间步，每个时间步一个值")
    numeric_columns: List[str] = Field(...)
    categorical_columns: List[str] = Field(default=[])
    embedding_configs: Optional[Dict[str, Dict]] = Field(default={},
                                                         description="分类列信息 {input_dim,input_length,output_dim,embeddings_regularizer}}")
    output_config: Dict[str, Dict] = Field(...,
                                           description="输出配置 {输出列: {type: regression/classification, ...}}")
    learning_rate: float = Field(default=0.00035)

    @classmethod
    def from_dict(cls, config: Optional[Dict]) -> Optional['MethodConfig']:
        if config is None:
            return None
        return MethodConfig(**config)

    @field_validator('output_config')
    def _validate_output_config(cls, v):
        for label_name, config in v.items():
            if not isinstance(config, dict):
                raise ValueError(f"输出配置 '{config}' 必须是字典")

            requirements = {'type', 'loss', 'metrics'}
            if not requirements.issubset(config.keys()):
                missing = requirements - set(config.keys())
                raise ValueError(f"配置缺少必需的字段: {missing}")
            if config['type'] not in ['regression', 'classification', 'binary_classification']:
                raise ValueError(f"输出类型必须是 'regression' 或 'classification' 或 'binary_classification'")

        return v


class SingleTaskLstmModel:
    """输出模式：单任务"""

    def __init__(self, config):
        set_global_seeds(config.get('seed', 42))
        self.model = None
        self.model_config = self._initialize_config(config)
        '''
        如果单层效果不好：可以加LSTM层数 units = [64, 32]  # 逐步压缩特征 return_sequences = [True, False]
        如果效果还不好，多层 压缩  1. 更深的网络 [128, 64, 32]  2. 更宽的网络 [64, 32] 
        '''

    def _initialize_config(self, config):
        if isinstance(config, dict):
            return MethodConfig.from_dict(config)
        elif isinstance(config, MethodConfig):
            return config
        else:
            error_msg = f"不符合要求的格式，config必须是支持的dict或者MethodConfig实例，实际类型: {type(config)}"
            raise ValueError(error_msg)

    def _build_lstm_model(self):
        input_width = self.model_config.input_width
        output_width = self.model_config.output_width
        num_cols = self.model_config.numeric_columns
        cat_cols = self.model_config.categorical_columns
        embedding_configs = self.model_config.embedding_configs
        output_config = self.model_config.output_config
        units = self.model_config.units
        return_sequences = self.model_config.return_sequences
        learning_rate = self.model_config.learning_rate

        numeric_input = tf.keras.layers.Input(
            shape=(input_width, len(num_cols)),
            name='numeric_input'
        )

        # 处理embedding层：无分类列直接跳过
        if cat_cols:
            categorical_inputs = []
            for col_name in cat_cols:
                cat_input = tf.keras.layers.Input(
                    shape=(input_width,),  # (6, ) 表示6个时间步，1个特征
                    name=f"categorical_{col_name}_input"
                )
                categorical_inputs.append(cat_input)

            # Embedding层处理分类特征
            embedded_layers = []
            for i, col_name in enumerate(cat_cols):
                embedding = tf.keras.layers.Embedding(**embedding_configs[col_name])(categorical_inputs[i])
                embedded_layers.append(embedding)
            if embedded_layers:
                if len(embedded_layers) > 1:
                    all_embedded = tf.keras.layers.Concatenate(axis=-1)(embedded_layers)
                else:
                    all_embedded = embedded_layers[0]
                combined = tf.keras.layers.Concatenate(axis=-1)([numeric_input, all_embedded])
            else:
                combined = numeric_input

            x = combined

        else:
            x = numeric_input
            categorical_inputs = []

        # 单层LSTM配置
        if len(units) == 1:
            x = tf.keras.layers.LSTM(units=units[0], return_sequences=return_sequences[0], activation='tanh',
                                     name='lstm_0')(
                x)
            x = tf.keras.layers.Dropout(0.2, seed=42)(x)

        # 多层LSTM(...ing)
        else:
            for i, (u, s) in enumerate(zip(units, return_sequences)):  # units列表长度代表 LSTM 层数
                x = tf.keras.layers.LSTM(units=u, return_sequences=s, activation='tanh', name=f'lstm_{i + 1}')(
                    x)

                # LayerNorm 归一化层
                x = tf.keras.layers.LayerNormalization(epsilon=1e-3, name=f'layernorm_{i}')(x)

            x = tf.keras.layers.Dropout(0.2)(x)

        outputs = []
        loss_dict = {}
        metric_dict = {}

        for output_name, config in output_config.items():

            num_label = len(output_name)

            # 回归任务: 输出形状 (batch_size, 5, 1)
            if config['type'] == 'regression':
                output_layer = tf.keras.layers.Dense(output_width * num_label,
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_width, num_label),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('linear', name=output_name)(output_layer)

                loss_dict[output_name] = config.get('loss', 'mse')
                metric_dict[output_name] = config.get('metrics', ['mae'])

            # 分类任务: 输出形状 (batch_size, 5, n_categories)
            elif config['type'] == 'classification':
                output_layer = tf.keras.layers.Dense(output_width * config['num_classes'],
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_width, config['num_classes']),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('softmax', name=output_name)(
                    output_layer)  # Keras模型输出名称由最后一个被命名的层决定

                loss_dict[output_name] = config.get('loss', 'sparse_categorical_crossentropy')
                metric_dict[output_name] = config.get('metrics', ['accuracy'])

            elif config['type'] == 'binary_classification':
                output_layer = tf.keras.layers.Dense(output_width * config['num_classes'],
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_width, config['num_classes']),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('sigmoid', name=output_name)(output_layer)
                loss_dict[output_name] = config.get('loss', 'binary_crossentropy')
                metric_dict[output_name] = config.get('metrics', ['accuracy'])

            else:
                output_layer = tf.keras.layers.Dense(output_width * num_label,
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_width, num_label),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('linear', name=output_name)(output_layer)

                loss_dict[output_name] = config.get('loss', 'mse')
                metric_dict[output_name] = config.get('metrics', ['mae'])

            outputs.append(output_layer)

        all_inputs = [numeric_input] + categorical_inputs  # 无分类数据时：[] ,有分类时：字典

        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_dict,
            loss_weights={'T': 1},
            metrics=metric_dict
        )

        return model


class MultiTasksLstmModel(SingleTaskLstmModel):
    """多任务输出"""

    def __init__(self, config):
        super().__init__(config)

    '''
            参数说明：
            output_config: 输出配置字典(每个输出特征单独一层)
            output_config = {
                        'temperature': {'type': 'regression', # 单变量回归
                                        'loss':'mse',
                                        'metrics':['mae'],
                                        'units': 1,  #  每个时间步预测n个特征
                                        },

                        'weather_metrics': {'type': 'regression', # 多变量回归：比如经度和纬度
                                            'loss':'mse',
                                            'metrics':['mae'],
                                            'units': 4,           # 每个时间步预测4个指标
                                            },

                        'event_occurrence': {'type': 'binary_classification', # 二分类
                                            'loss':'binary_crossentropy',
                                            'metrics':['accuracy'],
                                            'units': 1,
                                            },

                        'weather_type': {'type': 'classification', # 多分类
                                        'loss':'sparse_categorical_crossentropy',
                                        'metrics':['accuracy'],
                                        'num_classes': 3,
                                        },
                        }
    '''

    def _build_lstm_model(self):
        input_width = self.model_config.input_width
        output_width = self.model_config.output_width
        num_cols = self.model_config.numeric_columns
        cat_cols = self.model_config.categorical_columns
        embedding_configs = self.model_config.embedding_configs
        output_config = self.model_config.output_config
        learning_rate = self.model_config.learning_rate
        units = self.model_config.units  # len控制lstm的层数
        return_sequences = self.model_config.return_sequences  # 是否只在最后一个时间步产生输出，对应LSTM层数

        numeric_input = tf.keras.layers.Input(
            shape=(input_width, len(num_cols)),
            name='numeric_input'
        )

        # 处理embedding层：无分类列直接跳过
        if cat_cols:
            categorical_inputs = []
            for col_name in cat_cols:
                cat_input = tf.keras.layers.Input(
                    shape=(input_width,),  # (6, ) 表示6个时间步，1个特征
                    name=f"categorical_{col_name}_input"
                )
                categorical_inputs.append(cat_input)

            # Embedding层处理分类特征
            embedded_layers = []
            for i, col_name in enumerate(cat_cols):
                embedding = tf.keras.layers.Embedding(**embedding_configs[col_name])(categorical_inputs[i])
                embedded_layers.append(embedding)
            if embedded_layers:
                if len(embedded_layers) > 1:
                    all_embedded = tf.keras.layers.Concatenate(axis=-1)(embedded_layers)
                else:
                    all_embedded = embedded_layers[0]
                combined = tf.keras.layers.Concatenate(axis=-1)([numeric_input, all_embedded])
            else:
                combined = numeric_input

            x = combined

        else:
            x = numeric_input
            categorical_inputs = []

        """
        1.设置只在最后一个时间步产生输出:return_sequences=Fasle
        2.LSTM 层的参数总数【（64+19+1）*64】*4 == 【（上一轮输出+本轮输入）*（全联接输出）+（全连接输出层偏置）】*4层（遗忘门*1+记忆门*2+输出门*1）
          -a. 如果LSTM是第1层，那么输入就是(64+inputs.shape[1])个特征值。
          -b. 如果是后续层，接在另一个LSTM层之后(且前一层的return_sequences=True),那么输入维度将是前一层的输出维度 64,总输入=64+64=128
        """

        # 单层LSTM配置
        if len(units) == 1:
            x = tf.keras.layers.LSTM(units=units[0], return_sequences=return_sequences[0], activation='tanh',
                                     name='lstm_0')(
                x)
            x = tf.keras.layers.Dropout(0.2)(x)

        # 多层LSTM(...ing)
        else:
            for i, (u, s) in enumerate(zip(units, return_sequences)):  # units列表长度代表 LSTM 层数
                x = tf.keras.layers.LSTM(units=u, return_sequences=s, activation='tanh', dropout=0.1,
                                         recurrent_dropout=0.05, name=f'lstm_{i + 1}')(
                    x)

                # 可选：LayerNorm 归一化层
                x = tf.keras.layers.LayerNormalization(epsilon=1e-3, name=f'layernorm_{i}')(x)

            x = tf.keras.layers.Dropout(0.2)(x)

        # 多任务输出（每个单独一层）
        outputs =   {}
        loss_dict = {}
        metric_dict = {}
        loss_weights = {}

        for output_name, config in output_config.items():

            # 统一获取输出维度
            if config['type'] == 'classification':
                output_dim = config.get('num_classes', 1)  # 优先用 num_classes，没有则默认1
            else:
                output_dim = config.get('units', 1)

                # 回归任务: 输出形状 (batch_size, 5, 1)
            if config['type'] == 'regression':
                output_layer = tf.keras.layers.Dense(output_width * output_dim,
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_width, output_dim),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('linear', name=output_name)(output_layer)

                loss_dict[output_name] = config.get('loss', 'mse')
                metric_dict[output_name] = config.get('metrics', ['mae'])
                loss_weights[output_name] = config.get('loss_weights', 1)

            # 分类任务: 输出形状 (batch_size, 5, n_categories)
            elif config['type'] == 'classification':
                output_layer = tf.keras.layers.Dense(output_width * output_dim,
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_width, output_dim),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('softmax', name=output_name)(
                    output_layer)  # Keras模型输出名称由最后一个被命名的层决定

                loss_dict[output_name] = config.get('loss', 'sparse_categorical_crossentropy')
                metric_dict[output_name] = config.get('metrics', ['accuracy'])
                loss_weights[output_name] = config.get('loss_weights', 1)

            elif config['type'] == 'binary_classification':
                output_layer = tf.keras.layers.Dense(output_width * output_dim,
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_width, output_dim),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('sigmoid', name=output_name)(output_layer)
                loss_dict[output_name] = config.get('loss', 'binary_crossentropy')
                metric_dict[output_name] = config.get('metrics', ['accuracy'])
                loss_weights[output_name] = config.get('loss_weights', 1)

            else:
                output_layer = tf.keras.layers.Dense(output_width * output_dim,
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_width, output_dim),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('linear', name=output_name)(output_layer)

                loss_dict[output_name] = config.get('loss', 'mse')
                metric_dict[output_name] = config.get('metrics', ['mae'])
                loss_weights[output_name] = config.get('loss_weights', 1)

            outputs[output_name] = output_layer

        all_inputs = [numeric_input] + categorical_inputs  # 无分类数据时：[] ,有分类时：字典

        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics=metric_dict
        )

        return model
