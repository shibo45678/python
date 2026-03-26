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
    - 改写早停判断逻辑  check_save_mode :1 
      - min_gap_threshold=0.0015
      - gap_tolerance_ratio=1.3

    - min_delta: 1e-6 较小（几乎每个都判断gap)
### 2. 结果：
    inner_epoch 20
    loss: 0.0586 - mae: 0.1890 - val_loss: 0.0601 - val_mae: 0.1916 - learning_rate: 3.1321e-05
| inner_epoch | loss | val_loss | mae | val_mae | lr  |
|-------------|------|---|-----|---------|-----|
| 20          | 0.0586  | 0.0601  | 0.1890  |  0.1916 | 3.1321e-05 |



-----
### baseline3（inner_epoch 0-19）
开始时间：2026-03-08 00:00 

### 1. 配置：
### 参数

    - batch_size 32->16
    - learning_rate 0.0003
    - min_delta: 1e-4 

    - check_save_mode :2 
      - min_gap_threshold=0.002
      - gap_tolerance_ratio=1.07
    

### 2. 结果：
    inner_epoch 19
    loss: 0.0583 - mae: 0.1879 - val_loss: 0.0599⬇️ - val_mae: 0.1915⬇️ - learning_rate: 4.0909e-05

| inner_epoch | loss   | val_loss | mae    | val_mae | lr  |
|-------------|--------|----------|--------|---------|-----|
| 19          | 0.0583 | 0.0599   | 0.1879 | 0.1915  | 4.0909e-05|

----


### baseline4（inner_epoch 0-19）
开始时间：2026-03-16 00:00 

### 1. 配置：
### 1.1 特征工程

    - 增加
        - _add_wind_chill_index（风寒指数 方案B： 双门控）
        - _create_statistical_T（热惯性指数thermal_memory）
        - _add_atmospheric_stability_features（原有Tdew_diff、新增dew_suppression、boundary_layer_stability）
        - _add_thermal_inertia_features（thermal_inertia）
        - _add_radiation_features（energy_balance、night_cos、night_sin）
    
    - 删除
        - _create_statistical_T(气温分段 segments) 此为分类特征走 embedding

### 1.2 参数
    - 对应调整正则等组合方式：
        - cos_min_lr: 1e-6 -> 1e-5
        - 增加LSTM(层内dropout=0.015）
        - 增加clipnorm =10优化器梯度剪裁

### 1.3 数据集分布
    - 0.7:0.2:0.1（测试集有从最新数据截短）

### 2. 结果：
    inner_epoch 19
    loss: 0.05882⬆️ - mae: 0.1885 - val_loss: 0.05937⬇️ - val_mae: 0.19085⬇️ - learning_rate: 4.087e-05⬇️

| inner_epoch | loss   | val_loss | mae    | val_mae | lr  |
|-------------|--------|----------|--------|---------|-----|
| 19    | 0.0589 | 0.0594   | 0.1885 | 0.1909  | 4.087e-05|


    inner_epoch 28
    loss: 0.05752⬇️ - mae: 0.1866 - val_loss: 0.05927⬇️ - val_mae: 0.19063⬇️ - learning_rate: 3.5e-05⬇️

| inner_epoch | loss   | val_loss | mae    | val_mae | lr  |
|------------|--------|----------|--------|---------|-----|
| 28    | 0.0575 | 0.0593   | 0.1866 | 0.1906 | 3.5e-05|



----
### baseline5（inner_epoch 0-19）
开始时间：2026-03-23 00:00 

### 1. 配置：
### 1.1 特征工程
        - 滞后特征3，6小时
        -_add_atmospheric_stability_features（冷暖流）

### 2. 结果：
    inner_epoch 19
    loss: 0.0579 - mae: 0.1871 - val_loss: 0.0592 - val_mae: 0.1907⬆️ - learning_rate: 4.8707e-05

------


### 基于baseline4的继续训练 (inner_epoch 29-36)
开始时间：2026-03-17 00:00

### 1.配置
    - 更换学习率调度器 Force 
      . epoch <= 38: target_lr = 3.0e-05
      . epoch <= 43: target_lr = 2.5e-05
    - early_stopping : 10
    - total_epoch :100 

### 2.结果
    inner_epoch 36
    loss: 0.05650⬇️ - mae: 0.1848 - val_loss: 0.05927（微弱降低） - val_mae: 0.190545⬇️ - learning_rate: 3.0000e-05
| inner_epoch | loss   | val_loss | mae    | val_mae | lr  |
|-------------|--------|----------|--------|---------|-----|
| 36          | 0.0565| 0.0593  |0.1848| 0.1905 | 3.0e-05|~~~~










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


