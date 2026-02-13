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

    def __init__(self,
                 experiment_dir='/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm2#_20260208_154953/tf_checkpoints_stage2',
                 stage: int = 2):

        self.experiment_dir = experiment_dir
        self.stage = stage
        """
        experiment_dir :带时间戳+阶段stage 的基准模型目录
        stage：标明继续训练是哪个阶段keras
        """

    def load_latest_checkpoint(self, model):
        if self.experiment_dir is None:
            raise ValueError(f"继续训练的stage目录为空，检查配置continue_from是否为None")

        latest_epoch = 0
        latest_checkpoint = None

        for item in os.listdir(self.experiment_dir):
            if item.startswith('epoch_'):
                try:
                    epoch = int(re.search(r'epoch_(\d+)', item).group(1))  # 从0开始的轮数 ,取后一位为下一stage的起始位置

                    checkpoint = os.path.join(self.experiment_dir, f'epoch_{epoch}')
                    if epoch > latest_epoch and os.path.exists(
                            os.path.join(checkpoint, f'model_stage{self.stage}.keras')):  # 只在keras基础上更新
                        latest_epoch = epoch
                        latest_checkpoint = checkpoint
                except:
                    continue

        if latest_checkpoint:
            logger.debug(
                f"加载检查点 Epoch{latest_epoch + 1},已完成的第{latest_epoch + 1}次训练，内部epoch编码{latest_epoch} ")
            keras_file = os.path.join(latest_checkpoint, f'model_stage{self.stage}.keras')
            loaded_model = tf.keras.models.load_model(keras_file)
            model.set_weights(loaded_model.get_weights())  # 确保始终是同一个对象的引用

        return latest_epoch


class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, stage_number,
                 initial_epoch, metric='val_T_loss',
                 min_delta=1e-6):
        """
            Args:
                checkpoint_dir: 检查点保存目录 # 带tf_checkpoints_stage1
                stage_number: 阶段编号
                initial_epoch: 初始epoch（用户友好）
                metric: 监控的指标
                min_delta: 视为改进的最小变化量
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.stage_number = stage_number
        self.initial_epoch = initial_epoch  # 用户友好
        self.metric = metric
        self.min_delta = min_delta

        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.best_epoch = -1

        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):  # 内部友好
        """
        epoch: Keras传入的**当前实际内部epoch编号**
        当 fit(initial_epoch=19) 时：
           第1个epoch: epoch = 19
           第2个epoch: epoch = 20
           第3个epoch: epoch = 21
        """
        # 每个epoch结束时检查验证损失
        current_value = logs.get(self.metric, None)  # metrics dict_keys(['loss', 'mae', 'current_value', 'val_mae'])

        if current_value is None:
            logger.warning(f"指标{self.metric}不存在于logs中")
            return

        if current_value < self.best_val_loss - self.min_delta:
            self.best_val_loss = current_value
            self.best_epoch = epoch
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        current_epoch_dir = f'epoch_{epoch}/'  # 转成内部-内部
        for item in os.listdir(self.checkpoint_dir):
            item_path = os.path.join(self.checkpoint_dir, item)
            if (os.path.isdir(item_path) and item.startswith('epoch_')
                    and item != current_epoch_dir):  # 排除当前目录
                try:
                    import shutil
                    shutil.rmtree(item_path)
                except Exception as e:
                    logger.warning(f"删除旧检查点失败{item_path}:{e}")
        checkpoint_epoch_dir = os.path.join(self.checkpoint_dir, current_epoch_dir)  # 用目录格式并确保路径以斜杠结尾
        os.makedirs(checkpoint_epoch_dir, exist_ok=True)  # 必须创建目录 epoch_23

        # 保存新的检查点
        with self._suppress_output():
            # 1. 保存为.keras格式（用于predict以及继续训练）
            keras_path = os.path.join(checkpoint_epoch_dir,
                                      f'model_stage{self.stage_number}.keras')  # epoch下面的model.keras文件
            self.model.save(keras_path)  # 默认就是.keras格式

            # 2. 保存为SavedModel格式（用于部署）
            export_path = os.path.join(checkpoint_epoch_dir,
                                       f'saved_model_stage{self.stage_number}')  # epoch下面的saved_model文件夹
            self.model.export(export_path)

            # 3. 单独保存模型权重（model.keras已包含权重，此文件为兼容性/迁移学习保留）
            weight_path = os.path.join(checkpoint_epoch_dir, f'model_stage{self.stage_number}.weights.h5')
            self.model.save_weights(weight_path)

            # 4. 保存训练状态
            state_path = os.path.join(checkpoint_epoch_dir, f'training_state_stage{self.stage_number}.pkl')

        self.best_model_path = checkpoint_epoch_dir  # 直接赋值给外层变量
        logger.debug(
            f"\ninner_epoch {epoch}: 保存最佳模型到 {checkpoint_epoch_dir}, 监控指标：{self.metric}：{self.best_val_loss:.4f}")

    @staticmethod
    @contextlib.contextmanager
    def _suppress_output():
        """抑制模型保存时的输出"""
        old_stdout = sys.stdout  # 备份原来的"屏幕输出通道"
        old_stderr = sys.stderr  # 备份原来的"错误输出通道"
        sys.stdout = io.StringIO()  # 把输出重定向到一个"垃圾桶"（内存中的虚拟文件）
        sys.stderr = io.StringIO()

        try:
            yield

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def get_best_model_info(self):
        return {
            'best_epoch': self.best_epoch,
            'best_value': self.best_val_loss,
            'best_model_path': self.best_model_path
        }


class CosineAnnealingWarmRestarts(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr=0.00035, min_lr=1e-6, total_epochs=30, warmup_epochs=5,
                 warmup_power=2.0, restart_epochs: list = None):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_power = warmup_power
        self.restart_epochs = restart_epochs or []
        """
        参数:
        - initial_lr: 初始学习率 (0.00035)  -> 顶
        - min_lr: 最小学习率 (1e-6) ->  脚
        - total_epochs: 1周期的总epoch数
        - warmup_epochs: warmup阶段epoch数（先小学习率"热身"，再大学习率训练）
        - warmup_power: warmup曲线形状 (1=线性（直线）, 2=二次（曲线）)
        - restart_epochs: 重启点列表，如[15, 25]表示在第15、25个epoch重启
        """

    def optimal_cosine_annealing(self, epoch):
        """
        - epoch: 当前epoch
        """
        # 处理重启逻辑
        if self.restart_epochs and len(self.restart_epochs) > 0:
            restart_epochs = sorted(self.restart_epochs)
            current_cycle_start = 0
            cycle_length = self.total_epochs

            for i in range(len(restart_epochs)):
                restart_epoch = restart_epochs[i]
                if epoch >= restart_epoch:
                    current_cycle_start = restart_epoch

                    # 计算当前周期的长度
                    if i + 1 < len(restart_epochs):
                        next_restart = restart_epochs[i + 1]
                        cycle_length = next_restart - restart_epoch
                    else:
                        cycle_length = self.total_epochs - restart_epoch
                else:
                    # 处理第一个周期（0到第一个重启点）的情况
                    if i == 0:
                        cycle_length = restart_epoch - 0
                    break

            epoch_in_cycle = epoch - current_cycle_start
            effective_total = cycle_length
        else:
            epoch_in_cycle = epoch
            effective_total = self.total_epochs

        # Warmup阶段
        if epoch_in_cycle < self.warmup_epochs:
            # 非线性warmup: (epoch/warmup_epochs)^warmup_power
            # 语义：Epoch 1、2 < full LR，Epoch 3 = full LR → 设 warmup_epochs=3
            warmup_progress = (epoch_in_cycle + 1) / self.warmup_epochs  # 是当前周期内的相对位置 归一化到[0, 1]范围
            warmup_factor = warmup_progress ** self.warmup_power
            return self.min_lr + (self.initial_lr - self.min_lr) * warmup_factor  # 确保学习率始终大于最小

        if effective_total <= self.warmup_epochs:
            return self.min_lr

        # 余弦退火阶段
        adjusted_epoch = epoch_in_cycle - self.warmup_epochs
        adjusted_total = effective_total - self.warmup_epochs

        # 确保不除零
        if adjusted_total <= 0:
            return self.min_lr
        progress = adjusted_epoch / adjusted_total

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        # 可选：添加一点噪声防止卡在局部最优
        # noise_factor = 1.0 + 0.01 * np.random.randn()  # 1%的随机噪声
        # return min_lr + (initial_lr - min_lr) * cosine_decay * noise_factor

        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay


class ForceLRCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch=33): # 用户友好
        super().__init__()
        self.start_epoch = start_epoch

    def on_epoch_begin(self, epoch, logs=None):
        """
            epoch: Keras传入的当前实际epoch编号
            当 fit(initial_epoch=33) 时：
               第1个epoch: epoch = 33（从32到33）
               第2个epoch: epoch = 34
               第3个epoch: epoch = 35
        """
        if epoch <= 38:
            target_lr = 2.5e-05
        elif epoch <= 43:
            target_lr = 2.0e-05
        elif epoch <= 48:
            target_lr = 1.2e-05
        else:
            target_lr = 8e-05

        # 关键，修改现有优化器的学习率
        try:
            self.model.optimizer.learning_rate.assign(target_lr)
            logger.debug(f"Epoch_{epoch}: 强制设置LR = {target_lr:.2e}")

        except Exception as e:
            logger.debug(f"无法修改学习率: {e}")
            old = self.model.optimizer

            self.model.optimizer = tf.keras.optimizers.Adam(
                learning_rate=target_lr,
                beta_1=getattr(old, 'beta_1', 0.9),
                beta_2=getattr(old, 'beta_2', 0.999),
                epsilon=getattr(old, 'epsilon', 1e-7)
            )
            logger.debug(f"创建新优化器，但复制了超参数")


def TrainingSingleModel(model_name: str,
                        model,  # tf.keras.models
                        trainset,  # x,y
                        valset,
                        basic_dir,  # 目录
                        epochs: int = 20,  # 总轮数
                        verbose: int = 2,
                        early_stop_patience=5,
                        reduce_lr_patience=2,
                        continue_from_experiment: str = None
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
    os.makedirs(basic_dir, exist_ok=True)

    # 创建TF分片格式目录
    tf_checkpoint_dir = os.path.join(basic_dir, 'tf_checkpoints')
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
            val_loss = logs.get('val_T_loss', None)  # metrics dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

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
    log_dir = os.path.join(basic_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

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
                       basic_dir,
                       total_epochs: int = 20,
                       verbose: int = 2,
                       early_stop_patience=5,
                       min_delta=1e-6,
                       reduce_lr_patience=2,
                       monitor='val_T_loss',
                       continue_from_experiment: str = None,
                       ):
    """
    continue_from_experiment=None 首次训练 / 或带stage1的已存在keras文件目录
    continue_from_experiment='/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm2_20260206_222944/tf_checkpoints_stage1'"""

    if continue_from_experiment is None:
        os.makedirs(basic_dir, exist_ok=True)

        stage_number = 0
        inner_epoch = 0
        tf_checkpoint_stage_dir = os.path.join(basic_dir, 'tf_checkpoints_stage0')
        os.makedirs(tf_checkpoint_stage_dir, exist_ok=True)
        logger.info(f"首次训练模型，最佳模型将加载到:{tf_checkpoint_stage_dir}")
    # ============继续训练逻辑===============
    else:
        match = re.search(r"(.*)tf_checkpoints_stage(\d+)", str(continue_from_experiment))
        stage_number = int(match.group(2)) + 1
        tf_checkpoint_stage_dir = os.path.join(match.group(1), f'tf_checkpoints_stage{stage_number}')
        os.makedirs(tf_checkpoint_stage_dir, exist_ok=True)

        manager = SimpleTrainingManager(experiment_dir=continue_from_experiment, stage=stage_number - 1)
        inner_epoch = manager.load_latest_checkpoint(model)
        logger.debug(
            f"加载检查点 Epoch={inner_epoch + 1} 模型，已训练 {inner_epoch + 1}/{total_epochs} 轮。下一次训练 Epoch={inner_epoch + 2}")

    # ===========================================

    # 创建TensorBoard日志目录   训练后，bash 查看 tensorboard --logdir=~/Python/NeuralNetwork/weights/logs
    log_dir = os.path.join(tf_checkpoint_stage_dir, "board_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # """检查点"""
    # # 存储最佳模型信息
    # best_val_loss = float('inf')  # 初始化为正无穷大，每轮寻找最小值
    # best_model_path = None
    #
    # class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    #     def __init__(self, checkpoint_dir, stage_number, initial_epoch,metric='val_T_loss'):  # tf_checkpoints_stage1
    #         super().__init__()
    #         self.checkpoint_dir = checkpoint_dir
    #         self.stage_number = stage_number
    #         self.initial_epoch = initial_epoch # 用户友好
    #         self.metric = metric
    #
    #     def on_epoch_end(self, epoch, logs=None): # 内部友好(会
    #         nonlocal best_val_loss, best_model_path  # 不是局部变量，而是来自外层函数，但非全局作用域
    #
    #         # 每个epoch结束时检查验证损失
    #         val_T_loss = logs.get(self.metric, None)  # metrics dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])
    #
    #         if val_T_loss is not None and val_T_loss < best_val_loss-1e-6:
    #             best_val_loss = val_T_loss
    #
    #             # 删除旧的检查点目录
    #             import shutil
    #             current_epoch_dir = f'epoch_{epoch}/' # 转成内部-内部
    #             for item in os.listdir(self.checkpoint_dir):
    #                 item_path = os.path.join(self.checkpoint_dir, item)
    #                 if (os.path.isdir(item_path) and item.startswith('epoch_')
    #                         and item != current_epoch_dir):  # 排除当前目录
    #                     shutil.rmtree(item_path)
    #
    #             checkpoint_epoch_dir = os.path.join(self.checkpoint_dir, current_epoch_dir)  # 用目录格式并确保路径以斜杠结尾
    #             os.makedirs(checkpoint_epoch_dir, exist_ok=True)  # 必须创建目录 epoch_23
    #
    #             @contextlib.contextmanager
    #             def suppress_output():
    #                 old_stdout = sys.stdout  # 备份原来的"屏幕输出通道"
    #                 old_stderr = sys.stderr  # 备份原来的"错误输出通道"
    #
    #                 sys.stdout = io.StringIO()  # 把输出重定向到一个"垃圾桶"（内存中的虚拟文件）
    #                 sys.stderr = io.StringIO()
    #
    #                 try:
    #                     yield
    #
    #                 finally:
    #                     sys.stdout = old_stdout
    #                     sys.stderr = old_stderr
    #
    #             # 保存新的检查点（只保留一个最佳模型文件）
    #             with suppress_output():
    #                 # 1. 保存为.keras格式（用于predict以及继续训练）
    #                 keras_path = os.path.join(checkpoint_epoch_dir,
    #                                           f'model_stage{self.stage_number}.keras')  # epoch下面的model.keras文件
    #                 self.model.save(keras_path)  # 默认就是.keras格式
    #
    #                 # 2. 保存为SavedModel格式（用于部署）
    #                 export_path = os.path.join(checkpoint_epoch_dir, f'saved_model_stage{self.stage_number}') # epoch下面的saved_model文件夹
    #                 self.model.export(export_path)
    #
    #                 # 3. 单独保存模型权重（model.keras已包含权重，此文件为兼容性/迁移学习保留）
    #                 weight_path = os.path.join(checkpoint_epoch_dir, f'model_stage{self.stage_number}.weights.h5')
    #                 self.model.save_weights(weight_path)
    #
    #             best_model_path = checkpoint_epoch_dir  # 直接赋值给外层变量
    #             logger.debug(f"\ninner_epoch {epoch}: 保存最佳模型到 {checkpoint_epoch_dir}, {self.metric}={val_T_loss:.4f}")
    """自定义检查点"""
    checkpoint_callback = CustomCheckpointCallback(checkpoint_dir=tf_checkpoint_stage_dir,
                                                   stage_number=stage_number,
                                                   initial_epoch=inner_epoch + 1,
                                                   metric=monitor,
                                                   min_delta=min_delta)

    """学习率优化：首次余弦退火"""
    cosine_callback = CosineAnnealingWarmRestarts(
        initial_lr=0.00039,
        min_lr=1e-5,
        total_epochs=20,  # 1周期总轮数
        warmup_epochs=3,  # 4代表3轮热身 / 如果需要早停 耐心值至少是warmup_epochs的3-5倍
        warmup_power=2.0,
        restart_epochs=None)

    cosine_lr_optimal = tf.keras.callbacks.LearningRateScheduler(
        cosine_callback.optimal_cosine_annealing,
        verbose=1)

    """固定学习率"""
    force_callback = ForceLRCallback(start_epoch=inner_epoch + 1 if inner_epoch != 0 else 0,)

    """继续训练 更新余弦退火支持继续训练"""
    # def optimal_cosine_annealing_with_start(epoch, start_epoch=0, initial_lr=2.2584e-05,
    #                                         min_lr=1e-6, total_epochs=25, warmup_epochs=5):
    #     """支持从中间epoch开始的余弦退火"""
    #
    #     # 调整epoch：减去开始epoch
    #     adjusted_epoch = epoch + start_epoch
    #
    #     # 如果adjusted_epoch已经在warmup之后
    #     if adjusted_epoch >= warmup_epochs:
    #         # 直接进入余弦衰减阶段
    #         decay_epoch = adjusted_epoch - warmup_epochs
    #         decay_total = total_epochs - warmup_epochs
    #
    #         # 余弦衰减
    #         progress = decay_epoch / decay_total
    #         cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    #
    #         return min_lr + (initial_lr - min_lr) * cosine_decay
    #     else:
    #         # 还在warmup阶段（不太可能，因为你从27开始）
    #         progress = adjusted_epoch / warmup_epochs
    #         return min_lr + (initial_lr - min_lr) * (progress ** 2.0)
    #
    # # 使用
    # cosine_lr_optimal = LearningRateScheduler(
    #     lambda epoch: optimal_cosine_annealing_with_start(
    #         epoch,
    #         start_epoch=33,
    #         initial_lr=2.5e-05,
    #         min_lr=1e-6,
    #         total_epochs=25,
    #         warmup_epochs=0
    #     ),
    #     verbose=1
    # )

    """学习率优化：直接设置学习率的 Callback"""

    reduceLR_lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(  # ReduceLROnPlateau
        monitor=monitor,
        factor=0.5,  # 学习率减半 factor=0.7: 每次减30%（衰减更慢）
        patience=reduce_lr_patience,  # 2个epoch无改善就降低LR
        min_lr=1e-7,  # 最小学习率
        min_delta=min_delta,  # 需要更显著改善
        cooldown=0,
        verbose=2,
        mode='min')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,  # 整体验证损失
                                                      patience=early_stop_patience,  # 比学习率调度更耐心
                                                      mode='min',  # 当监测指标停止减少时训练停止（维持最小值）
                                                      min_delta=min_delta,  # 设置最小改善阈值，与ReduceLR相同  0.00001
                                                      restore_best_weights=True),
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    # ===========================================
    record = model.fit(
        trainset,
        validation_data=valset,
        epochs=total_epochs,  # 用户友好
        initial_epoch=inner_epoch + 1 if inner_epoch != 0 else 0,  # 用户友好 内部epoch=22（从0开始）这里填Epoch23 （Keras我已经完成了23轮训练）
        verbose=verbose,  # 设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录 2 epoch每轮输出一行记录
        callbacks=[
            # 1. 学习率：余弦退火
            cosine_lr_optimal,  # 替换ReduceLROnPlateau
            # force_callback,# 强制学习率
            # reduceLR_lr_scheduler,

            # 2. 早停：防止过拟合
            early_stopping,

            # 3. 最佳检查点
            checkpoint_callback,

            # 4. TensorBoard
            tensorboard_callback
        ]
    )

    # if best_model_path:
    #     if os.path.exists(f'{best_model_path}.index'):
    #         logger.debug(f"\n训练完成！最佳模型保存在: {best_model_path}")
    #
    #         # 验证文件
    #         if os.path.exists(f"{best_model_path}.index"):
    #             # 加载权重
    #             try:
    #                 model.load_weights(best_model_path)
    #             except Exception as e:
    #                 logger.debug(f"加载权重失败: {e}")
    #         else:
    #             logger.warning(f"警告：检查点文件不存在 {best_model_path}.index")
    #
    # else:
    #     logger.warning("\n未保存最佳模型文件")
    #     best_model_path = None

    save_dir = os.path.expanduser("~/Python/NeuralNetwork/temperature_forecasting/data/pics/")
    history_plot(history=record, model_name=model_name, save_dir=save_dir)
    best_model_info = checkpoint_callback.get_best_model_info()

    return record, best_model_info['best_model_path']  # epoch


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

"""
# 常见问题和调整方案：
问题                            | 解决方案
--------------------------------------------------------------
训练初期不稳定                  | 增加warmup_epochs (3→5)
前期收敛太慢                    | 减少warmup_epochs (5→3)或增加warmup_power (1.0→2.0)
后期还在下降，想继续训练         | 增加total_epochs (30→40)
后期过拟合                      | 减少total_epochs (30→25)或增加EarlyStopping
想尝试跳出局部最优              | 启用重启: restart_epochs=[15, 25]
"""
