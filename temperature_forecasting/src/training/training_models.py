import logging

logger = logging.getLogger(__name__)
from evaluation.model_visualization import history_plot
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def TrainingModel(model_name: str,
                  model,  # tf.keras.models
                  trainset,  # x,y
                  valset,
                  weights_dir,  # 目录
                  epochs: int = 20,  # 总轮数
                  verbose: int = 2,
                  ):
    trainset = trainset.map(lambda n, c, l: ((n, c), l))
    valset = valset.map(lambda n, c, l: ((n, c), l))

    # 确保权重保存目录存在
    os.makedirs(weights_dir, exist_ok=True)

    # 创建TF分片格式目录
    tf_checkpoint_dir = os.path.join(weights_dir, 'tf_checkpoints')
    os.makedirs(tf_checkpoint_dir, exist_ok=True)

    # 存储最佳模型信息
    best_val_loss = float('inf')  # 初始化为正无穷大，每轮寻找最小值
    best_model_path = None

    class CustomCheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, checkpoint_dir):
            super().__init__()
            self.checkpoint_dir = checkpoint_dir

        def on_epoch_end(self, epoch, logs=None):
            nonlocal best_val_loss, best_model_path  # 不是局部变量，而是来自外层函数，但非全局作用域

            # 每个epoch结束时检查验证损失
            val_loss = logs.get('val_loss',
                                None)  # 包含当前epoch的训练指标（metrics） dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

            # 如果当前可更新
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss

                # 清除旧的检查点文件
                for f in os.listdir(self.checkpoint_dir):
                    if f.startswith('ckpt_'):
                        os.remove(os.path.join(self.checkpoint_dir, f))

                checkpoint_prefix = os.path.join(self.checkpoint_dir,
                                                 'ckpt')  # 保存为TF分片格式- 使用model.save()而不是ModelCheckpoint

                # 保存新的检查点（只保留一个最佳模型文件）
                self.model.save(checkpoint_prefix, save_format='tf')  # 重点

                best_model_path = checkpoint_prefix  # 直接赋值给外层变量
                logger.debug(f"\nEpoch {epoch + 1}: 保存最佳TF分片模型到 {checkpoint_prefix}, val_loss={val_loss:.4f}")

    record = model.fit(
        trainset,
        validation_data=valset,
        epochs=epochs,
        verbose=verbose,  # 设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录 2 epoch每轮输出一行记录
        callbacks=[
            # 早停：防止过拟合
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # 监测指标
                                             patience=8,  # 没有进步的训练轮数，在这之后训练停止
                                             mode='min',  # 当监测指标停止减少时训练停止（维持最小值）
                                             min_delta=0.00001,  # 设置最小改善阈值
                                             restore_best_weights=True),

            # 模型检查点：使用分片格式保存权重(自定义 tf.keras.callbacks.ModelCheckpoint)
            CustomCheckpointCallback(tf_checkpoint_dir),

            # 添加学习率调度 提升训练效果
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # 学习率减半
                patience=3,  # 2个epoch无改善就降低LR
                min_lr=1e-7,  # 最小学习率
                verbose=2,
                mode='min'

            )
        ]
    )

    if best_model_path and os.path.exists(f'{best_model_path}.index'):
        logger.debug(f"\n训练完成！最佳模型保存在: {best_model_path}")

        # 验证文件
        if os.path.exists(f"{best_model_path}.index"):
            # 加载权重
            try:
                model.load_weights(best_model_path)
            except Exception as e:
                logger.debug(f"加载权重失败: {e}")
        else:
            logger.warning(f"警告：检查点文件不存在 {best_model_path}.index")

    else:
        logger.warning("\n未保存最佳模型文件")
        best_model_path = None

    history_plot(history=record, model_name=model_name)

    return record, best_model_path

# 一般训练规律 损失值：
# train loss 不断下降   validation loss不断下降---网络仍在学习
# train loss 不断下降   validation loss不断上升---网络过拟合，添加dropout和max pooling
# train loss 不断下降   validation loss趋于不变---网络欠拟合
# train loss 趋于不变   validation loss趋于不变---网络陷入瓶颈，减小学习率（自适应效果不大）和batch数量减少
# train loss 不断上升   validation loss不断上升---网络结构问题，训练超参数设置不当，数据集需要清洗等
# train loss 不断上升   validation loss不断下降---数据集有问题，建议重新选择


# 一般训练规律：准确度（整体训练趋势）
# train accuracy 不断上升   validation accuracy 不断上升---网络仍在学习
# train accuracy 不断上升   validation accuracy 不断下降---网络过拟合，添加dropout和max pooling
# train accuracy 不断上升   validation accuracy 趋于不变---网络欠拟合
# train accuracy 趋于不变   validation accuracy 趋于不变---网络陷入瓶颈，减小学习率（自适应效果不大）和batch数量减少
# train accuracy 不断下降   validation loss 不断下降---网络结构问题，训练超参数设置不当，数据集需要清洗等
# train accuracy 不断下降   validation loss 不断上升---数据集有问题，建议重新选择
