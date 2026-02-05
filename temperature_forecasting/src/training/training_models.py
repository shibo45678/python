import logging
import math

from keras.src.callbacks import LearningRateScheduler

logger = logging.getLogger(__name__)
from evaluation.model_visualization import history_plot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=全部显示, 1=隐藏INFO, 2=隐藏WARNING, 3=隐藏ERROR
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # 设置TensorFlow日志级别
tf.autograph.set_verbosity(0)  # 关闭AutoGraph详细日志

# 同时设置absl日志（TensorFlow使用的）
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import datetime
import contextlib
import sys
import io
import re

class SimpleTrainingManager:
    """继续训练管理器"""
    def __init__(self,experiment_dir=None):
        self.experiment_dir = experiment_dir

    def load_latest_checkpoint(self,model):
        if not self.experiment_dir:
            return 0
        checkpoint_dir = os.path.join(self.experiment_dir,'tf_checkpoints')
        if not os.path.exists(checkpoint_dir):
            return 0

        latest_epoch =0
        latest_checkpoint =None

        for item in os.listdir(checkpoint_dir):
            if item.startswith('model_epoch_'):
                try:
                    epoch = int(re.search(r'epoch_(\d+)',item).group(1))
                    checkpoint = os.path.join(checkpoint_dir,f'model_epoch_{epoch}')
                    if epoch > latest_epoch and os.path.exists(os.path.join(checkpoint,'model.keras')):
                        latest_epoch = epoch
                        latest_checkpoint = checkpoint
                except:
                    continue

        if latest_checkpoint:
            logger.debug(f"加载检查点 Epoch{latest_epoch} ")
            keras_file = os.path.join(latest_checkpoint,'model.keras')
            loaded_model = tf.keras.models.load_model(keras_file)
            model.set_weights(loaded_model.get_weights())

        return latest_epoch


def TrainingSingleModel(model_name: str,
                        model,  # tf.keras.models
                        trainset,  # x,y
                        valset,
                        weights_dir,  # 目录
                        epochs: int = 20,  # 总轮数
                        verbose: int = 2,
                        early_stop_patience=5,
                        reduce_lr_patience=2,
                        ):
    '''训练后，bash 查看 tensorboard --logdir=~/Python/NeuralNetwork/weights/logs'''

    # 处理数据形状
    def safe_map_function(x, y):
        """安全的 map 函数，确保返回有效数据"""
        if y is None:
            # 创建零值标签
            y = tf.zeros([tf.shape(x)[0], 5, 1], dtype=tf.float32)

        if isinstance(y, dict):
            # 从字典提取
            if 'T' in y:
                y = y['T']
            elif len(y) > 0:
                # 取第一个值
                y = list(y.values())[0]
            else:
                y = tf.zeros([tf.shape(x)[0], 5, 1], dtype=tf.float32)

        return x, y

    trainset = trainset.map(safe_map_function)
    valset = valset.map(safe_map_function)

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
            val_loss = logs.get('val_loss', None)  # metrics dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss

                # 删除旧的检查点目录
                import shutil
                current_epoch_dir = f'model_epoch_{epoch}/'
                for item in os.listdir(self.checkpoint_dir):
                    item_path = os.path.join(self.checkpoint_dir, item)
                    if (os.path.isdir(item_path) and item.startswith('model_epoch_')
                            and item != current_epoch_dir):  # 排除当前目录
                        shutil.rmtree(item_path)

                checkpoint_dir = os.path.join(self.checkpoint_dir, current_epoch_dir)  # 用目录格式并确保路径以斜杠结尾
                os.makedirs(checkpoint_dir, exist_ok=True)  # 必须创建目录

                # 保存为TF分片格式- 使用model.export()而不是ModelCheckpoint

                @contextlib.contextmanager
                def suppress_output():
                    old_stdout = sys.stdout  # 备份原来的"屏幕输出通道"
                    old_stderr = sys.stderr  # 备份原来的"错误输出通道"

                    sys.stdout = io.StringIO()  # 把输出重定向到一个"垃圾桶"（内存中的虚拟文件）
                    sys.stderr = io.StringIO()

                    try:
                        yield

                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

                # 保存新的检查点（只保留一个最佳模型文件）
                with suppress_output():
                    # 1. 保存为.keras格式（用于predict方法）
                    keras_path = os.path.join(checkpoint_dir, 'model.keras')
                    self.model.save(keras_path)  # 默认就是.keras格式

                    # 2. 保存为SavedModel格式（用于部署）
                    export_path = os.path.join(checkpoint_dir, 'saved_model')
                    self.model.export(export_path)

                best_model_path = checkpoint_dir  # 直接赋值给外层变量
                logger.debug(f"\nEpoch {epoch + 1}: 保存最佳模型到 {checkpoint_dir}, val_loss={val_loss:.4f}")

    # 创建TensorBoard日志目录
    log_dir = os.path.join(weights_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    record = model.fit(
        trainset,
        validation_data=valset,
        epochs=epochs,
        verbose=verbose,  # 设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录 2 epoch每轮输出一行记录
        callbacks=[
            # 早停：防止过拟合
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # 整体验证损失
                                             patience=early_stop_patience,  # 没有进步的训练轮数，在这之后训练停止
                                             mode='min',  # 当监测指标停止减少时训练停止（维持最小值）
                                             min_delta=0.0001,  # 设置最小改善阈值
                                             restore_best_weights=True),

            CustomCheckpointCallback(tf_checkpoint_dir),

            # 添加学习率调度 提升训练效果
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',  # 整体验证损失
                factor=0.5,  # 学习率减半
                patience=reduce_lr_patience,  # 2个epoch无改善就降低LR
                min_lr=1e-7,  # 最小学习率
                verbose=2,
                min_delta=0.0001,
                mode='min'
            ),

            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
        ]
    )

    if best_model_path:
        if os.path.exists(f'{best_model_path}.index'):
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

    save_dir = os.path.expanduser("~/Python/NeuralNetwork/temperature_forecasting/data/pics/")
    history_plot(history=record, model_name=model_name, save_dir=save_dir)

    return record, best_model_path


def TrainingMultiModel(model_name: str,
                       model,  # tf.keras.models
                       trainset,  # x,y
                       valset,
                       weights_dir,  # 目录
                       epochs: int = 20,  # 总轮数
                       verbose: int = 2,
                       early_stop_patience=5,
                       reduce_lr_patience=2,
                       continue_from_experiment: str = None
                       ):
    '''训练后，bash 查看 tensorboard --logdir=~/Python/NeuralNetwork/weights/logs'''

    # 确保权重保存目录存在
    os.makedirs(weights_dir, exist_ok=True)

    # 创建TF分片格式目录
    tf_checkpoint_dir = os.path.join(weights_dir, 'tf_checkpoints')
    os.makedirs(tf_checkpoint_dir, exist_ok=True)

    # ============继续训练逻辑===============
    initial_epoch = 0
    if continue_from_experiment:
        manager = SimpleTrainingManager(continue_from_experiment)
        initial_epoch = manager.load_latest_checkpoint(model)
        logger.debug(f"从Epoch {initial_epoch} 开始训练，总共 {epochs} 轮")
    # ===========================================

    # 存储最佳模型信息
    best_val_loss = float('inf')  # 初始化为正无穷大，每轮寻找最小值
    best_model_path = None
    """检查点"""
    class CustomCheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, checkpoint_dir):
            super().__init__()
            self.checkpoint_dir = checkpoint_dir

        def on_epoch_end(self, epoch, logs=None):
            nonlocal best_val_loss, best_model_path  # 不是局部变量，而是来自外层函数，但非全局作用域

            # 每个epoch结束时检查验证损失
            val_loss = logs.get('val_loss', None)  # metrics dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss

                # 删除旧的检查点目录
                import shutil
                current_epoch_dir = f'model_epoch_{epoch+1}/'
                for item in os.listdir(self.checkpoint_dir):
                    item_path = os.path.join(self.checkpoint_dir, item)
                    if (os.path.isdir(item_path) and item.startswith('model_epoch_')
                            and item != current_epoch_dir):  # 排除当前目录
                        shutil.rmtree(item_path)

                checkpoint_dir = os.path.join(self.checkpoint_dir, current_epoch_dir)  # 用目录格式并确保路径以斜杠结尾
                os.makedirs(checkpoint_dir, exist_ok=True)  # 必须创建目录

                @contextlib.contextmanager
                def suppress_output():
                    old_stdout = sys.stdout  # 备份原来的"屏幕输出通道"
                    old_stderr = sys.stderr  # 备份原来的"错误输出通道"

                    sys.stdout = io.StringIO()  # 把输出重定向到一个"垃圾桶"（内存中的虚拟文件）
                    sys.stderr = io.StringIO()

                    try:
                        yield

                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

                # 保存新的检查点（只保留一个最佳模型文件）
                with suppress_output():
                    # 1. 保存为.keras格式（用于predict方法）
                    keras_path = os.path.join(checkpoint_dir, 'model.keras')
                    self.model.save(keras_path)  # 默认就是.keras格式

                    # 2. 保存为SavedModel格式（用于部署）
                    export_path = os.path.join(checkpoint_dir, 'saved_model')
                    self.model.export(export_path)

                best_model_path = checkpoint_dir  # 直接赋值给外层变量
                logger.debug(f"\nEpoch {epoch + 1}: 保存最佳模型到 {checkpoint_dir}, val_loss={val_loss:.4f}")

    """学习率优化：余弦退火"""
    def optimal_cosine_annealing(epoch, initial_lr=0.00035, min_lr=1e-6,
                                 total_epochs=30, warmup_epochs=5,
                                 warmup_power=2.0, restart_epochs=None): # 在Epoch 20重启学习率
        """
        最优版余弦退火（带warmup，可选重启）

        参数:
        - epoch: 当前epoch
        - initial_lr: 初始学习率 (0.00035)
        - min_lr: 最小学习率 (1e-6)
        - total_epochs: 总epoch数
        - warmup_epochs: warmup阶段epoch数
        - warmup_power: warmup曲线形状 (1=线性, 2=二次)
        - restart_epochs: 重启点列表，如[15, 25]表示在第15、25个epoch重启
        """

        # 处理重启逻辑
        if restart_epochs:
            current_cycle_start = 0
            cycle_length = total_epochs

            for restart_epoch in sorted(restart_epochs):
                if epoch >= restart_epoch:
                    current_cycle_start = restart_epoch
                    cycle_length = total_epochs - restart_epoch
                else:
                    break

            epoch_in_cycle = epoch - current_cycle_start
            effective_total = cycle_length
        else:
            epoch_in_cycle = epoch
            effective_total = total_epochs

        # Warmup阶段
        if epoch_in_cycle < warmup_epochs:
            # 非线性warmup: (epoch/warmup_epochs)^warmup_power
            warmup_progress = (epoch_in_cycle + 1) / warmup_epochs
            warmup_factor = warmup_progress ** warmup_power
            return initial_lr * warmup_factor

        # 余弦退火阶段
        adjusted_epoch = epoch_in_cycle - warmup_epochs
        adjusted_total = effective_total - warmup_epochs

        # 确保不除零
        if adjusted_total <= 0:
            return min_lr

        cosine_decay = 0.5 * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total))

        # 可选：添加一点噪声防止卡在局部最优
        # noise_factor = 1.0 + 0.01 * np.random.randn()  # 1%的随机噪声
        # return min_lr + (initial_lr - min_lr) * cosine_decay * noise_factor

        return min_lr + (initial_lr - min_lr) * cosine_decay

    # 创建调度器
    cosine_lr_optimal = LearningRateScheduler(
        lambda epoch: optimal_cosine_annealing(
            epoch,
            initial_lr=0.00035,
            min_lr=1e-6,
            total_epochs=30,
            warmup_epochs=5,
            warmup_power=2.0,
            restart_epochs=None
        ),
        verbose=1)


    # 创建TensorBoard日志目录
    log_dir = os.path.join(weights_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



    record = model.fit(
        trainset,
        validation_data=valset,
        epochs=epochs,
        initial_epoch = initial_epoch, # 支持从指定epoch开始
        verbose=verbose,  # 设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录 2 epoch每轮输出一行记录
        callbacks=[

            cosine_lr_optimal,# 余弦退火替换ReduceLROnPlateau


            # 早停：防止过拟合
            tf.keras.callbacks.EarlyStopping(monitor='val_T_loss',  # 整体验证损失
                                             patience=early_stop_patience,  # 比学习率调度更耐心
                                             mode='min',  # 当监测指标停止减少时训练停止（维持最小值）
                                             min_delta=1e-4,  # 设置最小改善阈值，与ReduceLR相同  0.00001
                                             restore_best_weights=True),

            # 模型检查点：使用分片格式保存权重(自定义 tf.keras.callbacks.ModelCheckpoint)
            CustomCheckpointCallback(tf_checkpoint_dir),

            # # 添加学习率调度 提升训练效果
            # tf.keras.callbacks.ReduceLROnPlateau(
            #     monitor='val_T_loss',  # 整体验证损失
            #     factor=0.5,  # 学习率减半
            #     patience=reduce_lr_patience,  # 2个epoch无改善就降低LR
            #     min_lr=1e-7,  # 最小学习率
            #     min_delta=1e-4, #  需要更显著改善
            #     cooldown=0,
            #     verbose=2,
            #     mode='min'
            # ),



            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
        ]
    )

    if best_model_path:
        if os.path.exists(f'{best_model_path}.index'):
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

    save_dir = os.path.expanduser("~/Python/NeuralNetwork/temperature_forecasting/data/pics/")
    history_plot(history=record, model_name=model_name, save_dir=save_dir)

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


