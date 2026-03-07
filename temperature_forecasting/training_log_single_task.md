# 单任务 - 单层LSTM 

### baseline（inner_epoch 0-22）
开始时间：2026-03-06 00:00 

### 1. 配置：
### 1.1 特征处理
    - WeatherGenerationFromNumeric 
     - 1. 通用风速风向处理
     - 2. 统计特征：T相关（温度分段、滞后(1h,1d)）

### 1.2 参数
    - units: [256]
    - return_sequences': [False]
    - learning_rate: 0.0005 *
    - total_epochs: 50 *
    - early_stop_patience：10
    - min_delta: 1e-6

    - cosine_lr_optimal 配置
      - cos_min_lr:1e-5 *
      - cos_total_epochs：20 *
      - cos_warmup_epochs: 5  
      - warmup_power: 2.0
      - restart_epochs: None

    - batch_size: 32
    - optimizer: AdamW（weight_decay=1e-5）*
 
### 1.3 模型架构
    - LSTM(256, return_sequences=False)
    - Dropout(0.1)

    - Dense() 
    - Reshape()
    - Activation

### 2. 结果：
    inner_epoch 22
    loss: 0.0586 - mae: 0.1888 - val_loss: 0.0601 - val_mae: 0.1917 - learning_rate: 5.1290e-05
| inner_epoch | loss | val_loss | mae | val_mae | lr  |
|-------------|------|---|-----|---------|-----|
| 22 | 0.0586  | 0.0601  | 0.1888  |  0.1917 | 5.1290e-05 |



### baseline2（inner_epoch 0-20）
开始时间：2026-03-07 00:00 

### 1. 配置：
### 参数
    - 改写早停判断逻辑 
      - min_gap_threshold=0.0015
      - gap_tolerance_ratio=1.3
### 2. 结果：
    inner_epoch 20
    loss: 0.0586 - mae: 0.1890 - val_loss: 0.0601 - val_mae: 0.1916 - learning_rate: 3.1321e-05
| inner_epoch | loss | val_loss | mae | val_mae | lr  |
|-------------|------|---|-----|---------|-----|
| 20          | 0.0586  | 0.0601  | 0.1890  |  0.1916 | 3.1321e-05 |



### baseline2（inner_epoch 0-20）
开始时间：2026-03-08 00:00 

### 1. 配置：
### 1.1 特征工程
    - 风寒指数（风速和气温的交互关系）
    - 对应调整正则等组合方式：
        - 
        - 







# 单任务 - 多层LSTM 
### baseline（inner_epoch 1-22）
开始时间：2026-03-07 00:00 


    通常位置：LSTM -> LayerNorm -> Dropout -> (下一层 LSTM)
    
    LSTM(128, return_sequences=True, recurrent_dropout=0.1),   # 第一层返回序列
    LayerNormalization(),                                       # 归一化
    Dropout(0.2),                                               # 层间 Dropout

    LSTM(256, return_sequences=False, recurrent_dropout=0.1),  # 第二层返回最终输出
    LayerNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),                               # 可选的全连接层
    Dropout(0.2),
    Dense(1)   





- [ ] 待办事项示例
- [x] 待办事项示例
