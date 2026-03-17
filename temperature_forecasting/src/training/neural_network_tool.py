class ModelConfigManager:
    """统一管理模型配置的辅助类"""

    @staticmethod
    def get_loss_config(output_config):
        # 单和多输出，都是字典
        output_names = list(output_config.keys())
        loss_config = {}
        for output_name in output_names:
            cfg = output_config.get(output_name, {})
            loss_type = cfg.get('type', 'regression')
            loss_config[output_name] = cfg.get('loss', ModelConfigManager._get_default_loss(loss_type))
        return loss_config

    @staticmethod
    def get_metrics_config(output_config):

        output_names = list(output_config.keys())
        metrics_config = {}
        for output_name in output_names:
            cfg = output_config.get(output_name, {})
            loss_type = cfg.get('type', 'regression')
            metrics = cfg.get('metrics', ModelConfigManager._get_default_metrics(loss_type))
            metrics_config[output_name] = metrics if isinstance(metrics, list) else [metrics]

        return metrics_config

    @staticmethod
    def get_loss_weights_config(output_config):

        output_names = list(output_config.keys())
        loss_weights = {}
        for output_name in output_names:
            cfg = output_config.get(output_name, {})
            loss_weights[output_name] = cfg.get('loss_weights', 1.0)
        return loss_weights

    @staticmethod
    def _get_default_loss(loss_type):
        defaults = {
            'regression': 'mse',
            'classification': 'sparse_categorical_crossentropy',
            'binary_classification': 'binary_crossentropy'
        }
        return defaults.get(loss_type, 'mse')

    @staticmethod
    def _get_default_metrics(loss_type):
        defaults = {
            'regression': ['mae'],
            'classification': ['accuracy'],
            'binary_classification': ['accuracy']
        }
        return defaults.get(loss_type, ['mae'])
