# 实验日志：multi_lstm1 单层LSTM 

## baseline epoch 1-25
- 开始时间：2026-02-06 09:50
- 配置：
  - 模型
    - learning_rate: 0.00035,
    - units: [192], 
    - return_sequences': [False],
    - epochs': 30,
    - early_stop_patience：5,
    - loss_weights: 0.75:0.25
    
    - batch_size: 32
    - optimizer: Adam

    - cosine_lr_optimal 配置
      - initial_lr: 0.00035
      - min_lr: 1e-6
      - total_epochs：30
      - warmup: 5
      - warmup_power: 2.0
      - restart_epochs: None

- 结果：
- best_val_loss=0.0520 @ epoch 25
- T_loss: 0.0597 ，val_T_loss: 0.0601，T_mae: 0.1904 ，val_T_mae: 0.1919
- rh_loss: 0.0294，val_rh_loss: 0.0276，rh_mae: 0.1373，val_rh_mae: 0.1325
- learning_rate: 4.8295e-05




# 实验日志：multi_lstm2 双层LSTM 
## baseline epoch 1-27
- 开始时间：2026-02-06 14:40
- 配置：
  - 模型
    - learning_rate: 0.00035,
    - units: [192,96], 
    - return_sequences': [True,False],
    - epochs': 30,
    - **early_stop_patience：5**,
    - **loss_weights: 0.7:0.3**

    - batch_size: 32
    - optimizer: Adam

    - cosine_lr_optimal 配置
      - initial_lr: 0.00035
      - min_lr: 1e-6
      - total_epochs：30
      - warmup: 5
      - warmup_power: 2.0
      - restart_epochs: None

- 结果：
- best_val_loss=0.0514 @ epoch 27
- T_loss: 0.0618 ，val_T_loss: 0.0614，T_mae: 0.1940 ，val_T_mae: 0.1944
- rh_loss: 0.0302，val_rh_loss: 0.0278，rh_mae: 0.1391，val_rh_mae: 0.1345
- learning_rate: 2.2584e-05

## 第二阶段 epoch 28-33
- 修改dropout T任务：0.25 rh任务：0.1
- 使用optimal_cosine_annealing_with_start 继续训练
    - start_epoch=33,
    - initial_lr=2.5e-05,
    - total_epochs=25,
    - warmup_epochs=0
- 其他不变
-结果
- best_val_loss=0.0513 @ epoch 33
- T_loss: 0.0615⬇️ ，val_T_loss: 0.0613⬇️，T_mae: 0.1933⬇️ ，val_T_mae: 0.1942⬇️
- rh_loss: 0.0293⬇️，val_rh_loss: 0.0277⬇️，rh_mae: 0.1370⬇️，val_rh_mae: 0.1341⬇️
- learning_rate: 1.6387e-05 很低（后续需要调整）


## 第三阶段 epoch 34-39
- 增加 内部的dropout： dropout=0.1, recurrent_dropout=0.05
- 修改 dropout T任务：0.25 rh任务：0.0
- 修改：cosine_lr_optimal 替换成ForceLRCallback直接学习率的callback 2.2584e-05
- 其他不变

- 结果
  - best_val_loss=0.0512 @ epoch 39
  - T_loss: 0.0610⬇️ ，val_T_loss: 0.0612⬇️，T_mae: 0.1925⬇️ ，val_T_mae: 0.1943⬆️
  - rh_loss: 0.0285⬇️，val_rh_loss: 0.0278⬆️，rh_mae: 0.1350⬇️，val_rh_mae: 0.1340⬇️
  - learning_rate:2.5e-05（ForceLRCallback定的）


## 第四阶段 epoch 40
调整：
- T_dropout从0.25→0.26 
- 损失权重T 0.7:rh 0.3 → 0.65:0.35
- callback 学习率调整到学习率从2.5e-05降到2.0e-05 
- 
- 结果
  - **best_val_loss=0.0495**⬇️ @ epoch 40
  - T_loss: 0.0612⬆️ ，val_T_loss: 0.0611⬇️，T_mae: 0.1929⬆️ ，val_T_mae: 0.1941⬇️
  - rh_loss: 0.0284⬇️，val_rh_loss: 0.0278，rh_mae: 0.1348，val_rh_mae: 0.1339⬇️
  - learning_rate:2.00e-05（ForceLRCallback定的）


# 实验日志：multi_lstm2 双层LSTM （架构/特征工程改变）
# baseline 
- 开始时间：2026-02-06 23:40
- 配置：
  - 模型
    - learning_rate: 0.00035,
    - units: [192,96], 
    - return_sequences': [True,False],
    - epochs: 30,
    - early_stop_patience：5,
    ** loss_weights: 0.65:0.35**
    ** 内部的dropout： dropout=0.1, recurrent_dropout=0.05**
    ** T任务Dropout(0.27），rh任务Dropout(0.05)**

    - batch_size: 32
    - optimizer: Adam

    **- cosine_lr_optimal **
      - initial_lr: 0.00035
      - min_lr: 1e-6
      - total_epochs：30
      - warmup: 5
      - warmup_power: 2.0
      - restart_epochs: None
      - 
# 第一次架构 / 特征调整
- 开始时间：2026-02-06 23:40
  - 调整：
      - 增加滞后特征 T lag_1h,lag_1d,rh(yes)
      - 共享LSTM 256，专用LSTM（T=144, RH=96) return_sequences': [True] 专用层各自False
      - 专用dropout最佳：T_dropout=0.30, RH_dropout=0.05
      - 共享层：[0.10, recent 0.05]
      - early_stop_patience：8
      - reduce_lr_patience：3 （ReduceLROnPlateau）
      - epochs:50
      - loss_weights: 0.6:0.4

      - learning_rate: ReduceLROnPlateau
    
      ReduceLROnPlateau ：
          - lr: 0.00035
          - factor:0.5    
          - min_lr:1e-7
          - min_delta=1e-4
  
  - batch_size: 32
  - optimizer: Adam

结果：val_T_loss: 0.0618 epoch 20
T_loss: 0.0622 ,val_T_loss: 0.0618 | T_mae: 0.1939 ,val_T_mae: 0.1954
rh_loss: 0.0289 ,val_rh_loss: 0.0279 | rh_mae: 0.1353 ,val_rh_mae: 0.1334
loss: 0.0479 val_loss: 0.0483 
learning_rate: 8.7500e-05



# 继续训练
 - 调整
   - 弃用reduce_lr_patience：3 （ReduceLROnPlateau）改用余弦退火
   - optimizer: AdamW
   - 整体配置的改变
   - 尚未尝试修改batch，以及继续AdamW尝试 weight_decay=1e-5

   - 结果：
   val_T_loss: 0.0616 epoch 17 （未过拟合）
   T_loss: 0.0617 ,val_T_loss: 0.0616 | T_mae: 0.1933 ,val_T_mae: 0.1946
   rh_loss: 0.0291 ,val_rh_loss: 0.0276 | rh_mae: 0.1360 ,val_rh_mae: 0.1327
   loss: 0.0487 val_loss: 0.0480
   learning_rate: 2.1456e-05

   val_T_loss: 0.0616 epoch 22 （稍微过拟合）
   T_loss: 0.0613⬇️ ,val_T_loss: 0.0616⬇️| T_mae: 0.1933⬇️ ,val_T_mae: 0.1948⬇️
   rh_loss: 0.0289 ,val_rh_loss: 0.0276⬇️ | rh_mae:0.1356⬇️ ,val_rh_mae: 0.1327⬇️
   loss: 0.0484⬆️ val_loss: 0.0480⬇️
   learning_rate: 3.5455e-05⬇️
 - 
 - 贴上具体参数：
   multi_base_model_config = {'numeric_columns': num_cols,
                               'categorical_columns': cat_cols,
                               'time_column': time_col,
                               'input_width': 6,
                               'output_width': 5,
                               'shift': 24,

                               'total_epochs': 50,
                               'units': [256],  # 2:1 # len控制lstm的层数
                               'return_sequences': [True],  # 上一轮的输出做本轮输入input + 上一轮输出
                               'early_stop_patience': 5,
                               'min_delta': 1e-6,
                               'monitor': 'val_T_loss',
                               'verbose': 2,
                               
                               'learning_rate': 0.00039,
                               'cos_min_lr': 1e-5,
                               'cos_total_epochs': 20,
                               'cos_warmup_epochs': 1,  # 3代表进行2轮预测
                               # 'reduce_lr_patience': 3,
                               }

    multi_lstm_model_config2 = {**base_lstm_model_config, **{
        'model_type': 'multi_lstm2*', # 数字代表LSTM层数(包括：公共层和模型.py的专用LSTM）
        'multi_tasks': True,
        'output_config': {
            'T': {'type': 'regression',  # 单变量回归
                  'loss': 'mse',  # 主损失函数
                  'metrics': ['mae'],  # 额外指标：平均绝对误差
                  'loss_weights': 0.6,
                  'units': 1,  # 每个时间步预测n个特征
                  },

            'rh': {'type': 'regression',
                   'loss': 'mse',
                   'metrics': ['mae'],
                   'loss_weights': 0.4,
                   'units': 1}}
        
        # 首次训练配置：continue_from：None / 注意余弦配置
        'continue_from':'/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm2*_20260305_233439/tf_checkpoints_stage0',
        
        # 使用main.py继续训练（None)
        # 直接从continue_training.py加载训练完的最佳模型(带epoch的path)
        'final_best_model':'/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm2*_20260305_235745/tf_checkpoints_stage0/epoch_22'
    }}
