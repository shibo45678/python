import numpy as np
import os

os.environ['PYTHON_THREAD'] = 'child'  # 子线程（使用Agg后端不显示，但保存）
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def history_plot(history, save_dir, model_name=""):
    """通用训练历史绘图函数"""

    # 获取所有可用的指标
    available_metrics = list(history.history.keys())
    logger.debug(f"可用的指标: {available_metrics}")

    # 分离loss和其他指标
    # loss_metrics = [m for m in available_metrics if 'loss' in m and not m.startswith('val_')]
    # other_metrics = [m for m in available_metrics if 'loss' not in m and not m.startswith('val_')]

    loss_metrics = [m for m in available_metrics if m == 'loss']
    other_metrics = [m for m in available_metrics if m != 'loss' and not m.startswith('val_')]

    # 创建子图
    n_plots = 1 + len(other_metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

    if n_plots == 1:
        axes = [axes]

    # 绘制损失
    epochs = np.arange(1, len(history.history[loss_metrics[0]]) + 1)
    axes[0].plot(epochs, history.history[loss_metrics[0]], 'r-', label=f'Training {loss_metrics[0]}')
    if f'val_{loss_metrics[0]}' in history.history:
        axes[0].plot(epochs, history.history[f'val_{loss_metrics[0]}'], 'b-', label=f'Validation {loss_metrics[0]}')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制其他指标
    for i, metric in enumerate(other_metrics, 1):
        axes[i].plot(epochs, history.history[metric], 'g-', label=f'Training {metric}')
        if f'val_{metric}' in history.history:
            axes[i].plot(epochs, history.history[f'val_{metric}'], 'orange', label=f'Validation {metric}')
        axes[i].set_title(f'{model_name} - {metric}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()

    # 保存图表
    plot_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 重要：关闭图形释放内存
    logger.debug(f'图表已保存到：{plot_path}')

    return plot_path
