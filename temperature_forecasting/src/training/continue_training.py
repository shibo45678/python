from typing import Union
from utils.tensorflow_config import TensorFlowConfig
from models import MultiTasksLstmModel, MultiTasksCnnModel, SingleTaskLstmModel
from training.training_models import CustomCheckpointCallback, CosineAnnealingWarmRestarts, \
    ContinueCosineAnnealing, compile_for_continue, ForceLRCallback
import tensorflow as tf
import re
import cloudpickle
import pandas as pd
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def continue_training(
        pre_path='/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm2*_20260304_092539',
        checkpoint_dir: str = 'tf_checkpoints_stage1',
        continue_inner_epoch: int = 0,
        update_lr: Union[str, float] = 'from_history',
        cos_min_lr=1e-5,
        cos_total_epochs=20,
        cos_warmup_epochs=1,
        save_model_dir=None,  # 带stage
        total_epochs=50,
        early_stop_patience=5,
        min_delta=1e-6,
        monitor=None,
        verbose=2,
        check_save_mode=2,
        gap_tolerance_ratio=1.07,
        min_gap_threshold=0.002,
        weight_decay=1e-5,
        clipnorm=10,

):
    """
    Args:
        pre_path: 之前保存的预训练的数据和历史 部分材料， ？后续可自动搜索tf_checkpoints/epoch/keras文件
        continue_inner_epoch:从多少轮开始继续训练（不写可推，与文件夹名一致，epoch从0计数）
        update_lr = 'from_history'(从预训练结果获得）/ 'fixed'（根据轮次 固定学习率）/ 直接自定义值 0.00039
                    学习率调度器使用reduce和热火都需要（可不配），force的时候不用。
        save_model_dir = 保存新训练模型地址（不填默认在同目录下的加载配置的stage后面一个）
        cos_warmup_epochs:1代表热启动0轮，不预热直接进行衰减
        monitor:监控指标(单任务不需要提供monitor指标（多任务提供任务的名字，用字符串格式即可）

    """
    TensorFlowConfig.setup_environment()

    continue_dir = os.path.join(pre_path, 'continue_training')
    stage_dir = os.path.join(pre_path, checkpoint_dir)
    model_name = re.search(r'/([a-z]+_[a-z]+[\d]+.*?)_', pre_path).group(1)  # multi_lstm2*
    stage_number = int(re.search(r'(\d+)', checkpoint_dir).group(1))

    # 0.模型初始配置
    config_file = os.path.join(continue_dir, f'{model_name}_config_stage{stage_number}.cpkl')
    if not os.path.exists(config_file):
        config_file = os.path.join(continue_dir, f'{model_name}_config_stage0.cpkl')
        logger.debug(f"加载最佳阶段配置文件不存在，使用首轮配置完成参数提取{config_file}")
    with open(config_file, 'rb') as f:
        config = cloudpickle.load(f)

    output_config = config.get('output_config', {})

    # 1. 加载预处理数据
    train_save_path = os.path.join(continue_dir, f'{model_name}_preprocessed_data/train_dataset')
    val_save_path = os.path.join(continue_dir, f'{model_name}_preprocessed_data/val_dataset')
    trainset = tf.data.Dataset.load(train_save_path)
    valset = tf.data.Dataset.load(val_save_path)  # Hash 相同 -> 排除数据问题，问题在模型/优化器/GPU 非确定性（没数据顺序Shuffle 或 Load 问题）

    # 2. 加载该阶段 模型架构+权重（keras直接加载架构和权重load_model，h5才是加载load_model不行尝试重建+load_weights)
    logger.info(f"加载模型: {stage_dir}")
    try:
        latest_checkpoint_file = find_latest_checkpoint(stage_dir, extension='.keras')
        if latest_checkpoint_file:
            model = tf.keras.models.load_model(latest_checkpoint_file)
            logger.info(f'成功加载完整模型{latest_checkpoint_file}（架构+权重）')
        else:
            raise FileNotFoundError(f"在{stage_dir}中找不到.keras文件")
    except Exception as e:
        logger.warning(f".keras加载失败: {e}")
        model = reconstruct_model(model_name, config)
        latest_checkpoint_file = find_latest_checkpoint(stage_dir, extension='.weights.h5')
        if not latest_checkpoint_file:
            raise FileNotFoundError("也找不到weights.h5权重文件")
        model.load_weights(latest_checkpoint_file)

        # 编译器状态恢复
        opt = model.optimizer
        model = compile_for_continue(model=model, opt=opt, output_config=output_config)

        logger.info(f"重构模型结构 + 加载{latest_checkpoint_file}权重文件：{latest_checkpoint_file}")

    # 3. 确定初始轮数
    if continue_inner_epoch == 0:
        if latest_checkpoint_file is not None:
            continue_inner_epoch_ = infer_continue_epoch(latest_checkpoint_file)
        else:
            raise ValueError(f"continue_inner_epoch设置参数为0-自动推理。但未找到文件")
    else:
        continue_inner_epoch_ = continue_inner_epoch
    logger.info(
        f"从Epoch:{continue_inner_epoch_ + 1} 开始继续训练，inner_epoch:{continue_inner_epoch_},下一次训练 Epoch:{continue_inner_epoch_ + 2}")

    # 4. 设置初始学习率
    target_lr = get_initial_learning_rate(continue_inner_epoch_, stage_number=stage_number, dir=continue_dir,
                                          method=update_lr,
                                          model_name=model_name)
    logger.info(f"设置初始学习率为: {target_lr}")

    # 更新编译器(学习率/正则/裁剪） 旧优化器的动量状态丢失
    old_opt = model.optimizer
    old_config = old_opt.get_config()

    new_opt = tf.keras.optimizers.AdamW(
        learning_rate=target_lr,
        weight_decay=weight_decay,
        clipnorm=clipnorm,
        beta_1=old_config.get('beta_1', 0.9),
        beta_2=old_config.get('beta_2', 0.999),
        epsilon=old_config.get('epsilon', 1e-7)
    )
    model = compile_for_continue(model, new_opt, output_config)

    # 6. 设置新训练的模型保存目录
    if save_model_dir is None:
        save_model_dir = os.path.join(pre_path, f'tf_checkpoints_stage{stage_number + 1}')  # 没有指定保存目录，就直接取下一阶段
        os.makedirs(save_model_dir, exist_ok=True)

    # 7. 设置回调函数
    callbacks = get_continue_callbacks(
        checkpoint_model_dir=save_model_dir,
        stage_number=stage_number + 1,
        initial_epoch=continue_inner_epoch_ + 1,  # 用户友好
        metric=monitor,
        target_lr=target_lr,
        cos_min_lr=cos_min_lr,
        cos_total_epochs=cos_total_epochs,
        cos_warmup_epochs=cos_warmup_epochs,
        min_delta=min_delta,
        early_stop_patience=early_stop_patience,
        check_save_mode=check_save_mode, gap_tolerance_ratio=gap_tolerance_ratio, min_gap_threshold=min_gap_threshold,

    )

    # 8. 继续训练
    logger.info(f"继续训练Epoch: {continue_inner_epoch_ + 2}/{total_epochs} 轮")

    # import hashlib
    # import numpy as np
    # def get_model_fingerprint(model):
    #     hasher=hashlib.md5()
    #     for w in model.weights:
    #         hasher.update(w.numpy().tobytes())
    #     if hasattr(model,'optimizer') and model.optimizer:
    #         opt_vars=model.optimizer.variables
    #         for v in opt_vars:
    #             hasher.update(v.numpy().tobytes())
    #
    #     return hasher.hexdigest()
    # fp = get_model_fingerprint(model)
    # print(f"🔍 Model+Optimizer Fingerprint before fit: {fp}")
    #
    # tf.random.set_seed(42)
    # np.random.seed(42)
    # random.seed(42)
    # os.environ['PYTHONHASHSEED'] = '42'

    new_history = model.fit(
        trainset,
        validation_data=valset,
        epochs=total_epochs,  # 用户友好
        initial_epoch=continue_inner_epoch_ + 1 if continue_inner_epoch_ != 0 else 0,  # 用户友好
        verbose=verbose,  # epoch每轮输出一行记录
        callbacks=callbacks)

    # 9. 保存新阶段的训练历史
    history_path, csv_path = combine_training_history(new_history, continue_inner_epoch_, pre_stage=stage_number,
                                                      model_name=model_name,
                                                      save_dir=continue_dir)

    return model, history_path, csv_path, save_model_dir, new_history


def find_latest_checkpoint(stage_dir, extension: str = '.keras'):
    # 支持多个检查点存在的情况
    if not os.path.exists(stage_dir):
        return None

    checkpoint_files = []
    for root, dirs, files in os.walk(stage_dir):
        # root = '/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm2_20260206_222944'
        # dirs = ['tf_checkpoints','logs']  一次遍历只处理当前层级
        # files = []  # 顶层目录没有文件
        for file in files:
            if file.endswith(extension):
                checkpoint_files.append(os.path.join(root, file))

    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=os.path.getmtime, reverse=True)  # 修改时间

    return checkpoint_files[0]


def reconstruct_model(model_type, config):
    if model_type is None:
        raise ValueError("配置文件中缺少 model_type")

    if model_type.startswith('multi_lstm'):
        model_obj = MultiTasksLstmModel(config)
        clean_model = model_obj._build_lstm_model()

    elif model_type.startswith('multi_cnn'):
        model_obj = MultiTasksCnnModel(architecture_type='enhance_parallel', config=config)
        clean_model = model_obj._build_cnn_model()

    elif model_type.startswith('single_lstm'):
        model_obj = SingleTaskLstmModel(config)
        clean_model = model_obj._build_lstm_model()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return clean_model


def infer_continue_epoch(file_name):
    if file_name is not None:
        match = re.search(r'epoch_(\d+)', file_name, re.IGNORECASE)  # re.IGNORECASE 不区分大小写匹配
        if match is not None:  # Match 对象
            return int(match.group(1))
    logger.info('没有匹配到continue_inner_epoch_，使用默认值25')
    return 25  # 默认值


def get_initial_learning_rate(continue_inner_epoch_, dir, model_name, stage_number, method='from_history'):
    """
        智能获取初始学习率

        Args:
        continue_inner_epoch_: 起始epoch
        dir : continue_dir = '/Users/shibo/Python/NeuralNetwork/lstm..../continue_training'
        method= 'from_history' / 'fixed '固定
    """
    if type(method) == float:
        logger.debug(f'训练初始学习率获得方式:自定义数值{method}')
        return method

    else:
        if method == 'from_history':
            history_data = get_training_history(save_dir=dir, stage=stage_number, model_name=model_name)
            history_dict = history_data.get('history', None)
            lr_column = ['learning_rate', 'lr', 'LR', 'LearningRate']
            for col in lr_column:
                if col in history_dict and len(history_dict[col]) > continue_inner_epoch_:  # 内部
                    return history_dict[col][continue_inner_epoch_]  # 列表

        elif method == 'fixed':
            if continue_inner_epoch_ <= 38:
                return 2.5e-05
            elif continue_inner_epoch_ <= 43:
                return 2.0e-05
            elif continue_inner_epoch_ <= 48:
                return 1.2e-05
            else:
                return 8e-05
        else:
            logger.warning(f'未知继续训练初始学习率获得方式{method},使用默认值 1e-5')
            return 1e-5


def combine_training_history(new_history, continue_inner_epoch_, pre_stage, model_name, save_dir):
    history_path = os.path.join(save_dir, f'{model_name}_history_stage{pre_stage + 1}.cpkl')
    csv_path = os.path.join(save_dir, f'{model_name}_history_stage{pre_stage + 1}.csv')
    # history 是一个 Keras Histoy 对象 不可以直接dump
    # history.history：字典 / history.params：字典（可序列化） / history.epoch：列表（可序列化） 其他不可序列化
    # history.history:字典格式 {各个指标：[值的列表]}

    if hasattr(new_history, 'history'):
        new_history_dict = new_history.history if hasattr(new_history, 'history') else new_history
        new_epochs = new_history.epoch if hasattr(new_history, 'epoch') else list(
            range(len(new_history_dict.get('loss', []))))
    else:
        new_history_dict = new_history
        new_epochs = list(range(1, len(next(iter(new_history_dict.values())))))

    # 处理前次历史
    pre_history_data = get_training_history(save_dir=save_dir, model_name=model_name, stage=pre_stage)

    if not pre_history_data:
        logger.error(f"文件夹内找不到前次历史文件：{save_dir}")
        return None, None

    pre_history_dict = pre_history_data.get('history', {})
    pre_epochs = pre_history_data.get('epochs', [])
    pre_params = pre_history_data.get('params', {})

    if continue_inner_epoch_ < 0 or continue_inner_epoch_ >= len(pre_epochs):
        logger.warning(
            f"continue_inner_epoch_({continue_inner_epoch_})超出范围(0-{len(pre_epochs) - 1})，使用最后一个epoch")
        continue_inner_epoch_ = len(pre_epochs) - 1

    # 合并history_dict字典
    merged_history_dict = {}
    metrics = set(pre_history_dict.keys()) | set(new_history_dict.keys())  # 就要处理指标不一致的问题
    # 回调函数添加了新指标等：比如中途添加了EarlyStopping监控的指标

    for metric in metrics:
        pre_values = pre_history_dict.get(metric, [])

        if len(pre_values) > continue_inner_epoch_:
            truncated_pre_values = pre_values[:continue_inner_epoch_ + 1]
        else:
            truncated_pre_values = pre_values
        new_values = new_history_dict.get(metric, [])

        # 如果只有一边有该指标，用None填充另一边
        if metric not in pre_history_dict:
            truncated_pre_values = [None] * (continue_inner_epoch_ + 1)  # +1个[None,None,...]
        elif metric not in new_history_dict:
            new_values = [None] * len(new_epochs)

        merged_history_dict[metric] = truncated_pre_values + new_values

    # 合并epoch列表
    if len(pre_epochs) > continue_inner_epoch_:
        truncated_pre_epochs = pre_epochs[:continue_inner_epoch_ + 1]
    else:
        truncated_pre_epochs = pre_epochs

    merged_epochs = truncated_pre_epochs + new_epochs

    new_history_data = {
        'model_name': model_name,
        'history': merged_history_dict,
        'epochs': [int(e) for e in merged_epochs],
        'params': pre_params,  # 保留最开始的不变
        'stage': pre_stage + 1,
        'save_time': datetime.now().isoformat(),
        'merge_info': {
            'truncated_at_inner_epoch': continue_inner_epoch_,
            'truncated_at_actual_epoch': truncated_pre_epochs[-1] if truncated_pre_epochs else 0,
            'pre_history_length': len(pre_epochs),
            'new_history_length': len(new_epochs),
            'merged_length': len(merged_epochs)
        }
    }
    with open(history_path, 'wb') as f:
        cloudpickle.dump(new_history_data, f)

    df = pd.DataFrame(merged_history_dict)
    df.insert(0, 'epoch', merged_epochs)  # epoch=0
    df.to_csv(csv_path, index=False)
    logger.info(f"训练历史保存到: {csv_path}")
    return history_path, csv_path


def get_training_history(save_dir, model_name, stage):
    save_path = os.path.join(save_dir, f'{model_name}_history_stage{stage}.cpkl')

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            history_data = cloudpickle.load(f)
        return history_data


def get_continue_callbacks(checkpoint_model_dir, initial_epoch, metric, stage_number, target_lr, cos_min_lr,
                           cos_total_epochs, cos_warmup_epochs, min_delta,
                           early_stop_patience, check_save_mode, gap_tolerance_ratio, min_gap_threshold):
    callbacks = []

    # 1. ModelCheckpoint - 保存最佳模型
    checkpoint_callback = CustomCheckpointCallback(checkpoint_dir=checkpoint_model_dir,
                                                   stage_number=stage_number,
                                                   initial_epoch=initial_epoch,
                                                   metric=metric, min_delta=min_delta, patience=early_stop_patience,
                                                   check_save_mode=check_save_mode,
                                                   gap_tolerance_ratio=gap_tolerance_ratio,
                                                   min_gap_threshold=min_gap_threshold,
                                                   )

    callbacks.append(checkpoint_callback)

    #  2.学习率a-余弦退火-首次
    # if stage_number < 0:
    #     cosine_callback = CosineAnnealingWarmRestarts(
    #         initial_lr=target_lr,
    #         min_lr=cos_min_lr,
    #         total_epochs=cos_total_epochs,  # 1周期总轮数
    #         warmup_epochs=cos_warmup_epochs,  # 4代表3轮热身 / 如果需要早停 耐心值至少是warmup_epochs的3-5倍
    #         warmup_power=2.0,
    #         restart_epochs=None)
    #
    #     cosine_lr_optimal = tf.keras.callbacks.LearningRateScheduler(
    #         cosine_callback.optimal_cosine_annealing,
    #         verbose=1)
    #     logger.debug(
    #         f"首次训练-余弦退火：周期总轮数 {cos_total_epochs},热身轮数cos_warmup_epochs-1 {cos_warmup_epochs - 1}")
    #
    #     callbacks.append(cosine_lr_optimal)
    # else:
    #     # 2. 学习率余弦退火-继续训练
    #     cosine_callback = ContinueCosineAnnealing(
    #         initial_lr=target_lr,
    #         min_lr=cos_min_lr,
    #         total_epochs=cos_total_epochs,
    #         warmup_epochs=cos_warmup_epochs,  # 1代表0轮热身(不热身）
    #         warmup_power=2.0,
    #         start_epoch=initial_epoch)
    #     cosine_lr_optimal = tf.keras.callbacks.LearningRateScheduler(
    #         cosine_callback.optimal_cosine_annealing_with_start,
    #         verbose=1)
    #     logger.debug(
    #         f"继续训练-余弦退火：周期总轮数 {cos_total_epochs},热身轮数cos_warmup_epochs-1 {cos_warmup_epochs - 1}")
    #
    #     callbacks.append(cosine_lr_optimal)

    # 3.学习率b-固定值
    force_callback = ForceLRCallback()
    callbacks.append(force_callback)

    # 3.学习率c-plateau
    # reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if metric is None else f'val_{metric}_loss',
    #                                                        factor=0.5, # 0.7: 每次减30%（衰减更慢）
    #                                                        min_lr=1e-6,
    #                                                        patience=early_stop_patience, min_delta=min_delta, cooldown=0,
    #                                                        verbose=1, mode='min')
    #
    # callbacks.append(reduce_callback)

    # 4.TensorBoard日志（可选）
    log_dir = os.path.join(checkpoint_model_dir, "board_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)

    return callbacks


if __name__ == '__main__':
    import argparse  # 使用 argparse（用于bash调用）参数灵活不用改代码

    parser = argparse.ArgumentParser(description='继续训练模型')
    parser.add_argument('--pre_path', type=str, required=False, help='前次训练保存的model时间戳地址')
    parser.add_argument('--checkpoint_dir', type=str, default='tf_checkpoints_stage0', help='到stage的位置')
    parser.add_argument('--continue_inner_epoch', type=int, default=None,
                        help='可以手动输入内部epoch，也可以None自动查找最后的文件')
    parser.add_argument('--save_model_dir', type=str, default=None,
                        help='继续训练新模型的存储位置，带stage，不填None可以自动在checkpoint_dir处下推一个位置')
    parser.add_argument('--update_lr', type=str, default='from_history', help='默认从前次历史找最佳点的学习率')
    parser.add_argument('--monitor', type=str, default='val_loss', help='监控指标，默认是总损失')
    parser.add_argument('--verbose', type=int, default=2, help='2代表训练一次显示一行')

    args = parser.parse_args()

    model, _, _, save_model_dir, new_history = continue_training(
        pre_path='/Users/shibo/Python/NeuralNetwork/saved_model/single_lstm1_20260316_090043',
        checkpoint_dir='tf_checkpoints_stage0',
        continue_inner_epoch=0,  # 0自动推
        save_model_dir=None,  # 带stage
        update_lr=0.00015,  # float: 0.00035/ str : 'from_history'
        early_stop_patience=10,
        min_delta=1e-6,
        total_epochs=100,  # 包括了首次训练消耗的轮数
        cos_min_lr=1e-5,
        cos_total_epochs=20,
        cos_warmup_epochs=3,
        monitor=None,  # 'T',
        verbose=2,
        check_save_mode=2,
        gap_tolerance_ratio=1.07,
        min_gap_threshold=0.002,
        weight_decay=1e-5,
        clipnorm=10,

    )
    print(f"\n训练完成！相关信息保存在: {save_model_dir}")
    print(f"最终验证损失: {new_history.history['val_loss'][-1]:.4f}")
