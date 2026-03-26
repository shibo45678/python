import logging
import math
import random
import cloudpickle
import numpy as np
from .neural_network_tool import ModelConfigManager
from contextlib import contextmanager
logger = logging.getLogger(__name__)
from evaluation.model_visualization import history_plot
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=全部显示, 1=隐藏INFO, 2=隐藏WARNING, 3=隐藏ERROR
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # 设置TensorFlow日志级别
tf.autograph.set_verbosity(0)  # 关闭AutoGraph详细日志

import absl.logging  # 同时设置absl日志（TensorFlow使用的）

absl.logging.set_verbosity(absl.logging.ERROR)
import datetime
import contextlib
import sys
import io
import re


# from keras.src.callbacks import LearningRateScheduler


class TrainingMultiModel:
    def __init__(self, history_plot=True):
        self.history_plot = history_plot

    def training_model(self, model_name: str, trainset, valset,
                       learning_rate,
                       cos_min_lr, cos_total_epochs, cos_warmup_epochs,
                       total_epochs, verbose: int = 2, early_stop_patience=10,
                       check_save_mode=1,
                       gap_tolerance_ratio=1.3,
                       min_gap_threshold=0.0015,
                       output_config=None,
                       weight_decay=1e-5,
                       clipnorm=10,

                       monitor=None, min_delta=1e-6,
                       basic_dir: str = None,
                       continue_from: str = None,
                       model=None

                       # reduce_lr_patience=2, # Reduce学习率才用

                       ):
        """
        多任务模型：

        1. 首次训练：
            basic_dir !=None (带stage1的已存在keras文件目录)
            model 是构建的 self.training_model_（tf.keras.models）
            continue_from=None
        2. 继续训练：
            basic_dir 是None
            model 为空，函数内部加载load .keras文件来（不需要新构建）
            continue_from='/Users/shibo/AL/NeuralNetwork/saved_model/multi_lstm2_20260206_222944/tf_checkpoints_stage1'

        trainset, valset ：训练集，既包括X,也包括标签y
        early_stop_patience: 多少轮监控指标无变化就停止，融合在最佳模型类里，不用单独EarlyStopping
        monitor：最佳模型的监控指标（在单任务里不需要参数）
                None（默认）：多任务-代表监控总体验证损失val_loss（主），总提验证MAE val_mae(辅）
                str('T'):多任务-指定监控的某特征
        cos_total_epochs: 1周期余弦退火预热的总轮数
        cos_warmup_epochs：4代表3轮热身 / 如果需要早停 耐心值至少是warmup_epochs的3-5倍
        余弦退火（首次训练模式，支持restart_epochs重启，继续训练不支持，默认状态None）

        """

        if continue_from is None and basic_dir is not None and model is not None:
            os.makedirs(basic_dir, exist_ok=True)

            stage_number = 0
            inner_epoch = 0
            tf_checkpoint_stage_dir = os.path.join(basic_dir, 'tf_checkpoints_stage0')
            os.makedirs(tf_checkpoint_stage_dir, exist_ok=True)
            logger.info(f"首次训练模型，最佳模型将加载到:{tf_checkpoint_stage_dir}")

            model_ = model

            # 首次训练用 1. 余弦退火
            cosine_callback = CosineAnnealingWarmRestarts(
                initial_lr=learning_rate,
                min_lr=cos_min_lr,
                total_epochs=cos_total_epochs,
                warmup_epochs=cos_warmup_epochs,
                warmup_power=2.0,
                restart_epochs=None)

            cosine_lr_optimal = tf.keras.callbacks.LearningRateScheduler(
                cosine_callback.optimal_cosine_annealing,
                verbose=1)
            logger.debug(
                f"首次训练-余弦退火：周期总轮数 {cos_total_epochs},热身轮数cos_warmup_epochs-1： {cos_warmup_epochs - 1}")


        # ============继续训练逻辑===============
        else:
            match = re.search(r"(.*)tf_checkpoints_stage(\d+)", str(continue_from))
            stage_number = int(match.group(2)) + 1
            tf_checkpoint_stage_dir = os.path.join(match.group(1), f'tf_checkpoints_stage{stage_number}')
            os.makedirs(tf_checkpoint_stage_dir, exist_ok=True)

            manager = SimpleTrainingManager(continue_dir=continue_from,continue_stage=stage_number-1)
            inner_epoch, model_ = manager.load_latest_checkpoint()

            # 创建新的编译器
            old_opt = model_.optimizer
            old_config = old_opt.get_config()

            new_opt = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                clipnorm=clipnorm,
                beta_1=old_config.get('beta_1', 0.9),
                beta_2=old_config.get('beta_2', 0.999),
                epsilon=old_config.get('epsilon', 1e-7)
            )
            model_ = compile_for_continue(model=model_, opt=new_opt, output_config=output_config)

            logger.debug(
                f"加载检查点 Epoch={inner_epoch + 1} 模型，已训练 {inner_epoch + 1}/{total_epochs} 轮。下一次训练 Epoch={inner_epoch + 2}")

            # 继续训练用
            cosine_callback = ContinueCosineAnnealing(
                initial_lr=learning_rate,  # 继续训练手动修改在外层配置修改
                min_lr=cos_min_lr,
                total_epochs=cos_total_epochs,
                warmup_epochs=cos_warmup_epochs,
                warmup_power=2.0,
                start_epoch=inner_epoch + 1)
            cosine_lr_optimal = tf.keras.callbacks.LearningRateScheduler(
                cosine_callback.optimal_cosine_annealing_with_start,
                verbose=1)
            logger.debug(
                f"继续训练-余弦退火：周期总轮数 {cos_total_epochs},热身轮数cos_warmup_epochs-1： {cos_warmup_epochs - 1}")

        # callbacks===========================================

        """2. 自定义检查点"""
        checkpoint_callback = CustomCheckpointCallback(checkpoint_dir=tf_checkpoint_stage_dir,
                                                       stage_number=stage_number,
                                                       initial_epoch=inner_epoch + 1,
                                                       metric=monitor,
                                                       min_delta=min_delta,
                                                       patience=early_stop_patience,
                                                       check_save_mode=check_save_mode,
                                                       gap_tolerance_ratio=gap_tolerance_ratio,
                                                       min_gap_threshold=min_gap_threshold,
                                                       total_epochs = total_epochs)

        """学习率-固定 ForceLRCallback"""
        force_callback = ForceLRCallback()
        """ """
        """学习率-ReduceLROnPlateau"""

        reduceLR_lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=f'val_{monitor}_loss' if monitor is not None else 'val_loss',
            factor=0.5,  # 学习率减半 factor=0.7: 每次减30%（衰减更慢）
            patience=early_stop_patience,
            min_lr=1e-7,
            min_delta=min_delta,
            cooldown=0,
            verbose=2,
            mode='min')

        """3. TensorBoard日志目录"""
        # bash 查看 tensorboard --logdir ~/AL/NeuralNetwork/saved_model/single_lstm1_20260316_090043/tf_checkpoints_stage2/logs/
        log_dir = os.path.join(tf_checkpoint_stage_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )

        # callbacks===========================================

        # 诊断梯度
        # probe_gradient_norm(model_, trainset, num_batches=30)

        record = model_.fit(
            trainset,
            validation_data=valset,
            epochs=total_epochs,  # 用户友好
            initial_epoch=inner_epoch + 1 if inner_epoch != 0 else 0,
            # 用户友好 内部epoch=22（从0开始）这里填Epoch23 （Keras我已经完成了23轮训练）
            verbose=verbose,  # 设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录 2 epoch每轮输出一行记录
            callbacks=[
                # 1. 学习率：余弦退火
                cosine_lr_optimal,  # 替换ReduceLROnPlateau
                # force_callback, # 强制学习率
                # reduceLR_lr_scheduler,

                # 2. 最佳检查点（带早停）
                checkpoint_callback,
                # 3. TensorBoard
                tensorboard_callback
            ]
        )
        best_model_info = checkpoint_callback.get_best_model_info()

        if self.history_plot:
            save_dir = os.path.expanduser("~/AL/NeuralNetwork/temperature_forecasting/data/pics/")
            history_plot(history=record, model_name=model_name, save_dir=save_dir)

        return record, best_model_info['best_model_epoch_path']  # epoch


class TrainingSingleModel(TrainingMultiModel):
    def __init__(self, history_plot=False):
        super().__init__(history_plot=history_plot)

    def training_model(self, model_name: str, trainset, valset,
                       learning_rate,
                       cos_min_lr, cos_total_epochs, cos_warmup_epochs,
                       total_epochs: int = 20, verbose: int = 2, early_stop_patience=5,
                       check_save_mode=1,
                       gap_tolerance_ratio=1.3,
                       min_gap_threshold=0.0015,
                       monitor=None, min_delta=1e-6, output_config=None,
                       weight_decay=1e-5,
                       clipnorm=20,
                       basic_dir: str = None,
                       continue_from: str = None,
                       model=None
                       ):
        '''单任务模型：注释同 TrainingMultiModel'''

        # 处理数据形状
        def safe_map_function(x, y):
            """安全的 map 函数，确保返回有效数据"""
            if y is None:
                y = tf.zeros([tf.shape(x)[0], 5, 1], dtype=tf.float32)  # 创建零值标签

            if isinstance(y, dict):
                if 'T' in y:
                    y = y['T']
                elif len(y) > 0:
                    y = list(y.values())[0]  # 取第一个值（单特征的训练）
                else:
                    y = tf.zeros([tf.shape(x)[0], 5, 1], dtype=tf.float32)

            return x, y

        trainset = trainset.map(safe_map_function)
        valset = valset.map(safe_map_function)

        # 继续父类训练
        config = {'trainset': trainset, 'valset': valset,
                  'model_name': model_name,
                  'learning_rate': learning_rate,
                  'cos_min_lr': cos_min_lr,
                  'cos_total_epochs': cos_total_epochs,
                  'cos_warmup_epochs': cos_warmup_epochs,
                  'total_epochs': total_epochs,
                  'verbose': verbose, 'early_stop_patience': early_stop_patience,
                  'check_save_mode': check_save_mode,
                  'gap_tolerance_ratio': gap_tolerance_ratio,
                  'min_gap_threshold': min_gap_threshold,
                  'output_config': output_config,
                  'monitor': monitor, 'min_delta': min_delta,
                  'basic_dir': basic_dir,
                  'continue_from': continue_from,
                  'model': model}
        return super().training_model(**config)


class SimpleTrainingManager:
    """继续训练管理器"""

    def __init__(self,
                 continue_dir='/Users/shibo/AL/NeuralNetwork/saved_model/multi_lstm2#_20260208_154953/tf_checkpoints_stage0',
                 continue_stage: int = 0): # 上次

        self.continue_dir = continue_dir
        self.stage = continue_stage
        """
        experiment_dir :带时间戳+阶段stage 的基准模型目录
        stage：标明继续训练是哪个阶段keras
        """

    def load_latest_checkpoint(self):
        if self.continue_dir is None:
            raise ValueError(f"继续训练的stage目录为空，检查配置continue_from是否为None")

        latest_epoch = 0
        latest_checkpoint = None
        self.continue_dir_ =f"{self.continue_dir}/"

        for item in os.listdir(self.continue_dir_):
            if item.startswith('epoch_'):
                try:
                    epoch = int(re.search(r'epoch_(\d+)', item).group(1))  # 从0开始的轮数 ,取后一位为下一stage的起始位置

                    checkpoint = os.path.join(self.continue_dir_, f'epoch_{epoch}')
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
            loaded_model = tf.keras.models.load_model(keras_file)  # 有权重也有优化器状态
            return latest_epoch, loaded_model
        else:
            raise FileNotFoundError(
                f"继续训练，未找到.keras最佳模型文件，模型未成功加载\n")


class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, stage_number,
                 initial_epoch, metric=None,
                 min_delta=1e-6, patience=10, check_save_mode=1, gap_tolerance_ratio=1.3, min_gap_threshold=0.0015,
                 total_epochs=50
                 ):
        """
            Args:
                checkpoint_dir: 检查点新保存目录 # 带tf_checkpoints_stage1
                stage_number: 阶段编号
                initial_epoch: 初始epoch（用户友好）
                metric: None(单任务 or 多任务监控总指标），str 'T'(支持多任务指定任务监控）
                        标准都使用：“验证损失（主）”，“验证mae"/"损失gap过拟合间隙（辅）"判断。
                min_delta: 视为改进的最小变化量
                patience: 早停的耐心值

                check_save_mode:
                        1 - 验证损失满足min_delta条件的训练保存（min_delta高点）；在不满足条件里面的挑出来“虽然下降少，但gap比较好的”；
                            几乎每个都判断gap
                        2 - 验证损失满足大于等于min_delta的训练（min_delta较小，尽量不遗漏），需要再判断损失的val_loss - loss的gap是否最优。
                            同时判断当前mae与历史保存的mae。
                            涉及参数gap_tolerance_ratio：可接受的过拟合gap 倍率 ，指标损失差（val_loss - loss）*ratio
                            min_gap_threshold：防止gap为0的时候 allowed_gap 永远为0
                        模式1几乎每轮范围内的下降都判断gap，模式2大的下降直接纳入，只调整微弱
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.stage_number = stage_number
        self.initial_epoch = initial_epoch  # 用户友好
        self.metric = metric
        self.min_delta = min_delta
        self._ensure_generator()

        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_gap = min_gap_threshold
        self.best_model_path = None
        self.best_epoch = -1

        self.patience_counter = 0  # 耐心计数器
        self.stopped_epoch = 0  # 停止的epoch
        self.patience = patience
        self.check_save_mode = check_save_mode
        self.min_gap_threshold = min_gap_threshold
        self.gap_tolerance_ratio = gap_tolerance_ratio
        self.total_epochs=total_epochs

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _ensure_generator(self):
        try:
            tf.random.get_global_generator()
        except (RuntimeError, ValueError):
            tf.random.set_global_generator(tf.random.Generator.from_seed(42))

    def on_train_begin(self, logs=None):
        self._ensure_generator()

        if self.initial_epoch is not None and self.initial_epoch >= 0:  # 恢复前次stage的RNG
            match = re.search(r'(.+)tf_checkpoints_stage(\d+)', self.checkpoint_dir)
            pre_path = match.group(1)
            pre_stage = self.stage_number - 1
            rng_file = os.path.join(pre_path, f'tf_checkpoints_stage{pre_stage}',
                                    f'epoch_{self.initial_epoch - 1}', f'rng_stage{pre_stage}.cpkl')

            if os.path.exists(rng_file):
                with open(rng_file, 'rb') as f:
                    rng_state = cloudpickle.load(f)

                # 完全恢复状态，不仅仅是种子
                random.setstate(rng_state['python_random'])
                np.random.set_state(rng_state['numpy_random'])

                if rng_state['tf_random'] is not None:
                    new_gen = tf.random.Generator.from_state(rng_state['tf_random'], 'philox')
                    tf.random.set_global_generator(new_gen)
                logger.debug(f"恢复RNG状态，上次训练到epoch{rng_state['epoch']}")
            else:
                logger.debug(f"无RNG状态文件：{rng_file}")

            # 恢复早停状态
            training_state_file = os.path.join(pre_path, f'tf_checkpoints_stage{pre_stage}',
                                               f'epoch_{self.initial_epoch - 1}',
                                               f'training_state_stage{pre_stage}.cpkl')

            if os.path.exists(training_state_file):
                with open(training_state_file, 'rb') as f:
                    training_state = cloudpickle.load(f)
                self.best_val_loss = training_state['best_val_loss']
                self.best_val_mae = training_state['best_val_mae']
                self.best_epoch = training_state['best_epoch']
                self.patience_counter = 0
                logger.debug(f"恢复早停状态：best_loss={self.best_val_loss:.6f}, best_mae = {self.best_val_mae:.6f},"
                             f"best_epoch={self.best_epoch},patience_counter={self.patience_counter}。")

            # # 恢复学习率调度器状态（意外中断才用）
            # lr_state_file = os.path.join(pre_path,f'tf_checkpoint_stage{pre_stage}',
            #                              f'epoch_{self.initial_epoch -1}',
            #                              f'lr_schedule_stage{pre_stage}.cpkl')
            # if os.path.exists(lr_state_file):
            #     with open(lr_state_file,'rb') as f:
            #         lr_state = cloudpickle.load(f)
            #
            #         # 确保优化器的迭代次数正确
            #         if hasattr(self.model.optimizer,'iterations'):
            #             self.model.optimizer.iterations.assign(lr_state['optimizer_iterations'])
            #         logger.debug(f"恢复学习率状态：lr={lr_state['current_learning_rate']:.8f},"
            #                      f"iterations={lr_state['optimizer_iterations']}")

    def on_epoch_end(self, epoch, logs=None):  # 内部友好
        """
        epoch: Keras传入的**当前实际内部epoch编号**
        当 fit(initial_epoch=19) 时：
           第1个epoch: epoch = 19
           第2个epoch: epoch = 20
           第3个epoch: epoch = 21
        """

        # 每个epoch结束时检查验证损失（主）/ 验证MAE（辅）
        if self.metric is None:  # str 'T' 单任务 如果有可能是多任务也可能是单任务
            primary_key = 'val_loss'
            secondary_key = 'val_mae'
            tertiary_key = 'loss'
        else:
            if isinstance(self.metric, str):
                primary_key = f'val_{self.metric}_loss'
                secondary_key = f'val_{self.metric}_mae'
                tertiary_key = f'{self.metric}_loss'
            else:
                raise ValueError(f"多任务 monitor 参数， 目前只支持指定单个任务作为监控对象（str) ，不支持{self.metric}")

        current_val_loss = logs.get(primary_key, None)  # metrics dict
        current_val_mae = logs.get(secondary_key, None)
        current_loss = logs.get(tertiary_key, None)

        if current_val_loss is None or current_val_mae is None:
            logger.warning(f"指标{primary_key}不存在于logs中")
            return

        if self.check_save_mode == 1:

            if current_val_loss < self.best_val_loss - self.min_delta:

                current_gap = abs(current_val_loss - current_loss)
                allowed_gap = max(abs(self.best_gap) * self.gap_tolerance_ratio, self.min_gap_threshold)

                if current_gap <= allowed_gap:
                    self._update_best(current_val_loss, current_loss, current_val_mae, epoch)
                    self.best_gap = current_gap
                    self.patience_counter = 0
                    logger.debug(
                        f"改进！保存最佳模型，验证损失:{current_val_loss:.6f}，验证MAE：{current_val_mae:.6f}，"
                        f"损失GAP：{current_gap:.5f} ≤ {allowed_gap:.5f}")
                else:
                    self.patience_counter += 1
                    logger.debug(
                        f"跳过保存，patience_counter:{self.patience_counter}/{self.patience},"
                        f"验证损失下降但损失GAP - 当前gap {current_gap:.5f} > 最佳gap {allowed_gap:.5f}")
                    self._check_early_stop(epoch)
            else:
                self.patience_counter += 1
                logger.debug(f"最佳验证损失无改进，patience_counter:{self.patience_counter}/{self.patience}")
                self._check_early_stop(epoch)
        else:
            # 显著改进：验证损失下降等于或超过 min_delta]
            if self.best_val_loss - current_val_loss >= self.min_delta:
                self._update_best(current_val_loss, current_loss, current_val_mae, epoch)
                self.patience_counter = 0
                logger.debug(
                    f"显著改进！保存最佳模型，验证损失:{current_val_loss:.6f}，验证MAE：{current_val_mae:.6f}"
                    f"当前gap(val-train):{(current_val_loss - current_loss):.5f} ")

            # 微弱下降：下降量在 [0, min_delta) 之间
            elif self.best_val_loss - current_val_loss >= 0:

                current_gap = abs(current_val_loss - current_loss)
                allowed_gap = max(abs(self.best_gap) * self.gap_tolerance_ratio, self.min_gap_threshold)

                if current_gap <= allowed_gap and current_val_mae < self.best_val_mae:  # mae 也可以允许浮动一些
                    self._update_best(current_val_loss, current_loss, current_val_mae, epoch)
                    self.best_gap = current_gap
                    self.patience_counter = 0
                    logger.debug(
                        f"微弱改进！保存最佳模型，验证损失:{current_val_loss:.6f}，验证MAE：{current_val_mae:.6f}。"
                        f"当前gap:{current_gap:.5f} <= 允许gap:{allowed_gap:.5f}")

                else:
                    self.patience_counter += 1
                    logger.debug(
                        f"跳过保存（gap过大/mae大），patience_counter:{self.patience_counter}/{self.patience}，"
                        f"当前gap {current_gap:.5f} > 允许gap {allowed_gap:.5f}"
                        f"当前mae {current_val_mae:.5f} < 最佳mae {self.best_val_mae:.5f}")
                    self._check_early_stop(epoch)
            else:
                self.patience_counter += 1
                logger.debug(
                    f"最佳验证损失未下降，patience_counter:{self.patience_counter}/{self.patience} ")
                self._check_early_stop(epoch)

    def _check_early_stop(self, epoch):
        if self.patience_counter >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            logger.info(
                f"早停于 epoch {epoch}，最佳验证损失: {self.best_val_loss:.6f} ，"
                f"最佳验证MAE：{self.best_val_mae:.6f}。 ")
            return True
        return False

    def _update_best(self, current_val_loss, current_loss, current_val_mae, epoch):
        self.best_val_loss = current_val_loss
        self.best_loss = current_loss
        self.best_val_mae = current_val_mae
        self.best_epoch = epoch
        self.best_model_path = self._save_checkpoint(epoch)

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
            saved_path = os.path.join(checkpoint_epoch_dir,
                                       f'saved_model_stage{self.stage_number}')  # epoch下面的saved_model文件夹
            self.model.export(saved_path)

            # 3. 单独保存模型权重（model.keras已包含权重，此文件为兼容性/迁移学习保留）
            weight_path = os.path.join(checkpoint_epoch_dir, f'model_stage{self.stage_number}.weights.h5')
            self.model.save_weights(weight_path)

            # 4. 保存完整RNG状态
            tf_gen = tf.random.get_global_generator()
            rng_state = {
                'python_random': random.getstate(),
                'numpy_random': np.random.get_state(),
                'tf_random': tf_gen.state.numpy(),
                'epoch': epoch
            }
            rng_file = os.path.join(checkpoint_epoch_dir, f'rng_stage{self.stage_number}.cpkl')
            with open(rng_file, 'wb') as f:
                cloudpickle.dump(rng_state, f)

            # 5. 保存训练状态
            training_state = {
                'best_val_loss': self.best_val_loss,
                'best_val_mae': self.best_val_mae,
                'best_epoch': self.best_epoch,
                'patience_counter': self.patience_counter,
                'stopped_epoch': self.stopped_epoch,
                'stage_number': self.stage_number,
                'metric': self.metric,
                'min_delta': self.min_delta
            }
            state_path = os.path.join(checkpoint_epoch_dir, f'training_state_stage{self.stage_number}.cpkl')
            with open(state_path, 'wb') as f:
                cloudpickle.dump(training_state, f)

            # # 6. 保存学习率调度器状态（意外中断）
            # if  hasattr(self.model.optimizer,'learning_rate'):
            #     lr_schedule_state ={
            #         'current_learning_rate':float(self.model.optimizer.learning_rate.numpy()),
            #         'optimizer_iterations':int(self.model.optimizer.iterations.numpy())
            #     }
            #     lr_state_path = os.path.join(checkpoint_epoch_dir,
            #                                  f"lr_schedule_stage{self.stage_number}.cpkl")
            #     with open(lr_state_path,'wb') as f:
            #         cloudpickle.dump(lr_schedule_state,f)


        logger.debug(
            f"\ninner_epoch {epoch}: 保存最佳模型到 {checkpoint_epoch_dir}")

        return checkpoint_epoch_dir

    def on_train_end(self, logs=None):
        # 训练结束时打印早停信息
        if self.stopped_epoch > 0 :
            logger.info(f"早停于 inner_epoch {self.stopped_epoch}")
            logger.info(
                f"最佳模型 inner_epoch {self.best_epoch} ,最佳验证损失: {self.best_val_loss:.6f}，最佳验证MAE：{self.best_val_mae} ")

            self.get_best_model_path()


    @contextmanager
    def _suppress_output(self):
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
        # 没到早停，训练总轮数直接结束，不会走on_train_end，但是会走这里
        self.get_best_model_path()
        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'best_model_epoch_path': self.best_model_path,
            'patience_counter': self.patience_counter,
            'stopped_epoch': self.stopped_epoch
        }

    def get_best_model_path(self):
        if self.best_model_path is None:
            match = re.search(r'(.+)tf_checkpoints_stage(\d+)', self.checkpoint_dir)
            pre_path = match.group(1)
            pre_stage = self.stage_number - 1
            self.best_model_path = os.path.join(pre_path, f'tf_checkpoints_stage{pre_stage}',
                                                f'epoch_{self.initial_epoch - 1}')
            logger.info(f"本次训练没有生成最佳模型，最佳模型仍然是：上期{self.best_model_path}")



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
        - total_epochs: 1周期的总epoch数（warmup从最低到最高，然后最高降到最低。
                         学习率会继续按余弦函数的周期性演化，从而出现多个波峰和波谷（U形上升）。
                         如果不希望U形上升，将total_epochs 覆盖整个训练阶段）
        - warmup_epochs: warmup阶段epoch数（先小学习率"热身"，再大学习率训练）
        - warmup_power: warmup曲线形状 (1=线性（直线）, 2=二次（曲线）)
        - restart_epochs: 重启点列表，如[15, 25]表示inner_epoch, 在第16、26轮训练就是热身轮（inner_epoch=15，25）

        学习率调度（热身 + 完整余弦周期，即取余弦函数在 [0,2π] 上的完整波形）。学习率先下降后上升，形成一个 U 形。
        如果再重复，就会形成多个 U 形（波浪形）。
        a.如果重启会立刻打断原来的U上升，直接切为“warmup”顶到最高学习率。
        b.但学习率调度的total_epochs，warmup从最低到最高，然后最高降到最低。
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

        # 还在Warmup阶段
        if epoch_in_cycle + 1 < self.warmup_epochs:
            warmup_progress = (epoch_in_cycle + 1) / self.warmup_epochs  # 是当前周期内的相对位置 归一化到[0, 1]范围
            warmup_factor = warmup_progress ** self.warmup_power
            return self.min_lr + (self.initial_lr - self.min_lr) * warmup_factor  # 确保学习率始终大于最小

        if effective_total <= self.warmup_epochs:
            return self.min_lr

        # 余弦退火阶段
        adjusted_epoch = (epoch_in_cycle + 1) - (self.warmup_epochs - 1)
        adjusted_total = effective_total - (self.warmup_epochs - 1)

        # 确保不除零
        if adjusted_total <= 0:
            return self.min_lr

        progress = adjusted_epoch / adjusted_total

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay


class ContinueCosineAnnealing:
    """继续训练 更新余弦退火支持继续训练"""

    def __init__(self, initial_lr=0.00035, min_lr=1e-6, total_epochs=25, warmup_epochs=1,
                 warmup_power=2.0, start_epoch=23):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_power = warmup_power
        self.start_epoch = start_epoch  # 用户友好

    def optimal_cosine_annealing_with_start(self, epoch):
        """支持从中间epoch开始的余弦退火"""
        """
        参数说明:
            - epoch: 当前全局 epoch
            - start_epoch: 继续训练的起始 Epoch (实际是用户友好)
            - warmup_epochs: 真实的热身轮数 (1 代表无热身，2 代表热身 1 轮)，与首次余弦退火对应
            """

        # 调整epoch：减去开始epoch
        adjusted_epoch = epoch - self.start_epoch + 1

        # 如果adjusted_epoch已经在warmup之后
        if adjusted_epoch >= self.warmup_epochs:
            # 直接进入余弦衰减阶段
            decay_epoch = adjusted_epoch - (self.warmup_epochs - 1)
            decay_total = self.total_epochs - self.warmup_epochs + 1

            # 余弦衰减
            progress = decay_epoch / decay_total
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        else:
            # 还在warmup阶段
            progress = adjusted_epoch / self.warmup_epochs
            return self.min_lr + (self.initial_lr - self.min_lr) * (progress ** self.warmup_power)


class ForceLRCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        """epoch: Keras传入的当前实际内部epoch编号"""
        if epoch <= 38:
            target_lr = 3.0e-05
        elif epoch <= 43:
            target_lr = 2.5e-05
        elif epoch <= 48:
            target_lr = 2.0e-05
        else:
            target_lr = 8e-05

        # 关键，修改现有优化器的学习率
        try:
            self.model.optimizer.learning_rate.assign(target_lr)
            logger.debug(f"Epoch_{epoch+1}: 强制设置LR = {target_lr:.2e}")

        except Exception as e:
            logger.debug(f"ForceLRCallback训练过程中，无法修改学习率: {e}")

    def on_epoch_end(self,epoch,logs=None):
        # 进度条加学习率
        logs = logs or {} # logs 是 Keras 内部维护的一个字典

        current_lr = self.model.optimizer.learning_rate # 不同优化器获取学习率方式可能不同
        # 获取到的是张量或变量（调度器对象本身）要变成 数值numpy()
        if hasattr(current_lr,'numpy'):
            current_lr = current_lr.numpy()
        logs['learning_rate'] = current_lr


def probe_gradient_norm(model, dataset, num_batches=10):
    logger.debug("\n 开始探测梯度范数 (Probe Mode)...")

    # 确保模型已编译
    if not model.optimizer:
        raise ValueError("模型必须先 compile 才能探测梯度！")

    grad_norms = []

    if hasattr(dataset, '__iter__'):
        data_iter = iter(dataset)
    else:
        # 如果是 numpy 数组，简单处理一下 (假设是 (x, y) 元组)
        raise TypeError("请传入 tf.data.Dataset 对象用于探测")

    for i in range(num_batches):
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            logger.debug("数据集长度不足，提前结束探测。")
            break

        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            # 使用模型编译时指定的 loss 函数
            loss = model.compiled_loss(y_batch, predictions, regularization_losses=model.losses)

        # 计算梯度
        grads = tape.gradient(loss, model.trainable_weights)
        grads = [g for g in grads if g is not None]

        if grads:
            g_norm = tf.linalg.global_norm(grads)
            val = g_norm.numpy()
            grad_norms.append(val)
            logger.debug(f"  Batch {i}: 梯度范数 = {val:.4f}")
        else:
            logger.debug(f"  Batch {i}: 无梯度 (检查 Loss 或输入)")

    if grad_norms:
        avg_norm = sum(grad_norms) / len(grad_norms)
        max_norm = max(grad_norms)
        logger.debug("-" * 30)
        logger.debug(f"探测结果汇总:")
        logger.debug(f"   平均范数: {avg_norm:.4f}")
        logger.debug(f"   最大范数: {max_norm:.4f}")

        suggested_clip = max(1.0, max_norm * 1.2)  # 留 20% 余量
        logger.debug(f"    建议设置 clipnorm = {suggested_clip:.1f}")
        logger.debug("-" * 30)
        return suggested_clip
    else:
        logger.debug("未能计算出有效梯度。")
        return None


def compile_for_continue(model, opt, output_config):  # 同一Python进程中直接获取实例。独立的演化路径
    """为预测模型重新编译 多输出会折叠metrics会折叠"""

    # 使用统一的配置管理器
    loss_config = ModelConfigManager.get_loss_config(output_config)
    metrics_config = ModelConfigManager.get_metrics_config(output_config)
    loss_weights_config = ModelConfigManager.get_loss_weights_config(output_config)

    logger.debug("=== 编译前检查 ===")
    logger.debug(f"loss_config: {loss_config}")
    logger.debug(f"loss_config类型: {type(loss_config)}")
    logger.debug(f"metrics_: {metrics_config}")
    logger.debug(f"metrics类型: {type(metrics_config)}")
    logger.debug(f"loss_weights:{loss_weights_config}")
    logger.debug(f"loss_weights类型:{type(loss_weights_config)}")

    model.compile(
        optimizer=opt,
        loss=loss_config,  # 字典 键是输出层名
        loss_weights=loss_weights_config,
        metrics=metrics_config
    )

    logger.debug("编译完成，验证metrics配置...")
    if len(model.metrics) >= 2:
        compile_metrics = model.metrics[1]
        if hasattr(compile_metrics, '_user_metrics'):
            actual_metrics = compile_metrics._user_metrics
            logger.debug(f"实际编译的metrics配置: {actual_metrics}")
            logger.debug(f"期望的metrics配置: {metrics_config}")
    return model
