# 优化方向：
# 追加参数检测 window / handle_extre_numeric
# process_categorical_cols 支持排除列
# 时间列处理中新生成列 独立放入feature_generator
# 每种类型里面 生成新的特征 模块点 小标题系统点（风矢量是数值列、process将encoding onehot 纳入、feature generator 可以独立、时间列（单）处理和生成分离
# 模型output_config是单列配置，优化成同一类配置可以一起跑。什么都不写默认回归？分类？
# 缺失值填充 增加算法填充等
# missing列logger 从提示改为报错 尽早暴露/结果无意义/
# 统一下划线/标准化列有不存在需要提前验证
# config.yaml
# 线程内，实现窗口和模型拆分，主要是继续训练需要避免再做一次预处理（线程不包括预处理，fit里面除了模型训练还有窗口数据形成， 尽量不要线程外保存一次，内再保存一次）
# 多任务的LSTM未实现 注意力机制等 / cnn的单独训练
# 测试集小(保证连续）可能会导致测试集的指标不好，因为回测的数据代表性不足，分布不一致等
# 处理缺失增加自定义函数 'custom':{'columns':[],'func':partial()}, 目前是类
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging.config

import pandas as pd

from logging_config import LOGGING_CONFIG
import logging

logger = logging.getLogger(__name__)

from src.data.data_preprocessing.data_splitting import TimeSeriesSplitter, SplitByTimepoints
from src.data.data_preprocessing.data_sampling import SimpleTimeSampler
from src.data.exploration import VisualizationForNeural
from src.data.data_preparation.load_data import DataLoader
from src.data.data_preparation.describe_data import DescribeData
from src.data.data_preparation.remove_duplicates import RemoveDuplicates
from src.data.data_preparation.delete_cols import DeleteUselessCols
from src.data.data_preparation.fix_problem_cols import ProblemColumnsFixed, SpecialColumnsFixed
from src.data.data_preparation.check_extre_numeric_features import CheckExtreFeatures
from src.data.data_preparation.handle_extre_numeric_features import StatisticsOutlierDetector, NumericOutlierProcessor, \
    detect_, handle_
from src.data.data_preparation.handle_extre_categorical_features import CategoricalOutlierProcessor
from src.data.data_preparation.handle_missing_values import NumericMissingValueHandler, CategoricalMissingValueHandler
from src.data.data_preparation.identify_cols_type import ColumnsTypeIdentify
from src.data.data_preparation.convert_categorical_features import ConvertCategoricalColumns
from src.data.data_preparation.convert_numeric_features import ConvertNumericColumns
from src.data.data_preparation.analyze_time_series_gaps import ProcessTimeseriesColumns, ProcessContinuous
from src.data.data_preparation.handle_missing_values import RemoveNanHandler, HistNanHandler
from src.data.feature_engineering.feature_generation_from_numeric import WeatherGenerationFromNumeric
from src.data.feature_engineering.feature_generation_from_timecol import GenerationFromTimeseries
from src.data.feature_engineering.feature_selection import BasedOnCorrSelector
from src.data.feature_engineering.feature_scaling import UnifiedFeatureScaler
from src.data.feature_engineering.feature_encoding import OrdinalCategoricalEncoder
from src.data.feature_engineering.feature_transformer import CustomTransformer
from src.deployment.model_deployer import DeploymentManager
from src.utils.tensorflow_config import TensorFlowConfig
from src.training.neural_network_controller import TimeSeriesEstimator
from src.trained.model_postprocessor import TimeSeriesPostProcessor, MetricsCalculator
from src.pipelines.preprocess_pipeline import CompletePreprocessor


def main():
    logging.config.dictConfig(LOGGING_CONFIG)
    TensorFlowConfig.setup_environment()
    '''
    1. 严格按照数据的【处理顺序】使用‘class’，并标记'len_change'(这里将改变数据长度的步骤，手动处理）
    2. 手动处理的类:是无法放进pipeline的类，不会继承BaseEstimator和TransfromerMixin。并且使用learn_process处理。
    3. 一次训练直接 predict（keras格式），
       隔日预测 predict_with_best_checkpoint(new_data)（keras格式） ，
       专用部署 save + load(格式Saved Model)
    4.config2多层LSTM，config单层LSTM
    '''
    check_outliers_config = {'method': 'iqr', 'threshold': 1.5}
    download_duplicates_config = {
        'enabled': True,
        'path': '~/AL/NeuralNetwork/temperature_forecasting/data/intermediate',
        'filename': 'duplicate_rows.csv'}
    download_outliers_details_config = {
        'enabled': True,
        'path': '~/AL/NeuralNetwork/temperature_forecasting/data/intermediate',
        'filename': 'outliers.csv'}

    detector = StatisticsOutlierDetector(zscore_outlier_threshold=3.0, iqr_outlier_threshold=1.5)
    numeric_outliers_config = {'detect_and_handle_config': [
        # {'iqr': {'columns': ['p', 'VPmax', 'VPact', 'VPdef', 'T', 'Tpot', 'Tdew', 'rh', 'sh', 'H2OC', 'rho', 'wd'],
        #          'handle_method': ['clip', 'clip', 'clip', 'clip', 'clip', 'clip', 'clip', 'clip', 'clip', 'clip',
        #                            'clip', 'clip'], 'threshold': 1.5}},
        {'custom': {'columns': ['wv', 'max. wv'], 'handle_method': ['custom', 'custom'],
                    'detect_function': partial(detect_, threshold=0),
                    'handle_function': partial(handle_)}},
        # {'isolationforest': {'columns': [], 'handle_method': [], 'contamination': 0.025,
        #                      'random_state': 42}},
        # {'zscore': {'columns': [], 'handle_method': [], 'threshold': 3}},
        {'custom': {'columns': ['auto_handle_remain'], 'handle_method': ['ignore'], 'skip_handle': [],
                    'detect_function': partial(detector.recommend_detection_method)}}
    ],
        'generate_outlier_indicator': [],  # 选填 label
    }

    categorical_outliers_config = {
        'rare_threshold': 0.01,
        'similarity_threshold': 0.8,
        'auto_correct_typos': True}

    numeric_missing_config = {
        'spec_fill': [
            {'constant': {'columns': ['wv', 'max. wv'], 'fill_value': [0, 0]}},
            {'mode': {'columns': []}},
            {'ffill': {'columns': []}},  # 气压变化相对连续稳定，短期内有持续性
            {'bfill': {'columns': ['rh']}}  # 湿度变化相对缓慢，受天气系统影响有持续性
        ],
        'skip_fill': [],
        'smart_fill_remain': True,
        'important_columns': ['T'],
    }

    # 新特征生成后
    scaling_config = {
        'transformers': [
            {'standard': {'columns': []}},
            # 'p', , 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh', 'H2OC', 'rho', 'wv', 'max. wv', 'wd'
            {'minmax': {'columns': ['rh'], 'feature_range': (0, 1)}},  # rh有界变量（0～100），MinMax 天然匹配
            # 相同方法，相同其他参数配置，在columns列表填写
            {'minmax': {'columns': [], 'feature_range': (-1, 1)}},  # 相同方法，但是其他参数配置与前一配置不同，允许在下一行填写
            {'robust': {'columns': ['T', 'wv_y', 'max. wv_x', 'max. wv_y'], 'quantile_range': (25, 75)}}
        ],
        'skip_scale': ['is_night', 'is_cold_front', 'hour_sin', 'hour_cos', 'day_of_year_cos', 'day_of_year_sin',
                       'Season_sin',
                       'Season_cos']  # 跳过二分类列(数值型）/ 异常值标记列 / missing标记

    }

    # 1. 加载数据
    loader = DataLoader(input_files=['data_climate.csv'], pattern="new_*.csv", data_dir='data/raw')
    raw_data = loader.learn_process()

    # 2. 数据集分割流水线（数据质量：调整时间序列的连续性 + 历史数据填充大片时间序列 + 数据集分割）
    data_split_configs = [
        {'obj_list': [RemoveDuplicates(pass_through=False),
                      ProcessContinuous(interactive=False, create_extract_continuous=True)], 'len_change': True},
        {'obj_list': [HistNanHandler(lookback_years=10, pass_through=False)], 'len_change': False},
        {'obj_list': [SplitByTimepoints(val_start='2014-01-01 00:10:00', test_start='2016-01-01 00:10:00',
                                        column_name='Date Time')], 'len_change': True}]
    # 3. 预处理流水线(生成训练、验证、预测数据)
    preparation_configs = [
        {'obj_list': [DescribeData(), DeleteUselessCols()], 'len_change': False},
        {'obj_list': [RemoveDuplicates(download_config=download_duplicates_config, pass_through=True)],
         'len_change': True},
        {'obj_list': [ColumnsTypeIdentify(),
                      ProcessTimeseriesColumns(interactive=False, pass_through=True),
                      ConvertCategoricalColumns(categorical_columns=[]),
                      ConvertNumericColumns(preserve_object_integer_types=True, exclude_cols=['Date Time']),
                      ProblemColumnsFixed(problem_columns=['wv']), SpecialColumnsFixed(problem_columns=['T']),  # wv 一样
                      CheckExtreFeatures(method_config=check_outliers_config,
                                         download_config=download_outliers_details_config),

                      NumericMissingValueHandler(method_config=numeric_missing_config, pass_through=False),
                      CategoricalMissingValueHandler(method_config=None, pass_through=True),  # 后续隔离森林精细去异常

                      NumericOutlierProcessor(method_config=numeric_outliers_config),  # iqr / 业务初筛 --> 异常值初筛
                      CategoricalOutlierProcessor(method_config=categorical_outliers_config, strategy='consolidate')
                      ], 'len_change': False},

        {'obj_list': [SimpleTimeSampler(time_column='Date Time', freq_hours=1, minute=0, second=0)],
         'len_change': True},

        {'obj_list': [GenerationFromTimeseries(time_column='Date Time', plot=False),
                      WeatherGenerationFromNumeric(
                          selected_columns=['wd', 'wv', 'max. wv', 'T', 'rh', 'Tdew', 'VPdef', 'hour_sin', 'is_night'],
                          create_statistical=True, create_interactions=True)],
         'len_change': False},

        {'obj_list': [RemoveNanHandler(pass_through=False)], 'len_change': True},

        {'obj_list': [BasedOnCorrSelector(pass_through=True),
                      CustomTransformer(model_name='lstm', pass_through=False,
                                        power_columns=['rh', 'wv_y', 'max. wv_y'],
                                        power_method='yeo-johnson', power_standardize=False,
                                        power_skip=['hour_cos', 'Season_cos', 'day_of_year_sin', 'day_of_year_cos',
                                                    'hour_sin', 'is_night', 'Season_sin'],
                                        asinh_columns=[], asinh_skip=[],
                                        asinh_scale_factor=1.0),
                      UnifiedFeatureScaler(method_config=scaling_config, algorithm='lstm'),  # 自动根据数据分布及算法类型进行推荐标准化
                      OrdinalCategoricalEncoder(encode_order_cols={
                          'segments': ['极寒', '严寒', '寒冷', '冰点下', '低温', '凉', '舒适', '暖', '热']},
                          handle_unknown='use_encoded_value', unknown_value=-1, pass_through=True),
                      VisualizationForNeural(pass_through=True),
                      ], 'len_change': False},
    ]

    splitter = CompletePreprocessor(data_split_configs)
    (df_train, df_val, df_test), _ = splitter.train(features=raw_data, labels=None)
    logger.info(f"训练集数：{len(df_train)}，验证集数:{len(df_val)}，测试集数：{len(df_test)}。")
    time_col = splitter._get_step(0, 1).valid_time_column_


    preprocessor = CompletePreprocessor(preparation_configs)
    features_temp_train, _ = preprocessor.train(features=df_train, labels=None)

    features_temp_val, _ = preprocessor.transform_predict(features=df_val, labels=None)
    features_temp_test, _ = preprocessor.transform_predict(features=df_test, labels=None)

    num_cols = preprocessor.get_specific_attribute(6, 'engineer_2', 'numeric_columns_')  # 取第7个class的第4步的属性
    cat_cols = preprocessor.get_specific_attribute(6, 'engineer_3', 'categorical_columns_')

    # 4. 并行模型训练、评估
    base_lstm_model_config = {'numeric_columns': num_cols,
                              'categorical_columns': cat_cols,
                              'time_column': time_col,
                              'input_width': 6,
                              'output_width': 5,
                              'shift': 24,
                              'batch_size': 16,

                              'units': [256],  # len控制lstm的层数
                              'return_sequences': [False],  # 最后一层才是False,注意与分任务的LSTM衔接
                              'verbose': 2,
                              'total_epochs': 50,

                              'early_stop_patience': 15,
                              'check_save_mode': 2,
                              'gap_tolerance_ratio': 1.07,
                              'min_gap_threshold': 0.002,

                              'min_delta': 1e-4,  # 1e-4
                              'learning_rate': 0.0003,
                              'cos_min_lr': 1e-5,
                              'cos_total_epochs': 25,
                              'cos_warmup_epochs': 5,  # 首次训练支持restart

                              'weight_decay': 1e-5,
                              'clipnorm': 10,  # 首次fit前先诊断 只拦截 >10.0 的异常尖峰
                              }

    single_lstm_model_config1 = {**base_lstm_model_config, **{
        # 集中切换: 单/多任务 and 继续训练
        'model_type': 'single_lstm1',  # 数字代表LSTM层数(包括：公共层和模型.py的专用LSTM）
        'multi_tasks': False,
        'compute_feature_importance': False,
        'output_config': {
            'T': {'type': 'regression',
                  'loss': 'mse',
                  'metrics': ['mae'],
                  'loss_weights': 1,
                  'units': 1,
                  }},
        # 直接main.py文件继续训练（至stage)
        'continue_from': None,
        # '/Users/shibo/AL/NeuralNetwork/saved_model/single_lstm1_20260316_090043/tf_checkpoints_stage2',

        # continue_training.py文件的继续训练结果（至epoch)
        'final_best_model': None,
        # '/Users/shibo/AL/NeuralNetwork/saved_model/single_lstm1_20260316_090043/tf_checkpoints_stage2/epoch_37'
    }}

    # multi_lstm_model_config2 = {**base_lstm_model_config, **{
    #     'model_type': 'multi_lstm2*',  # 数字代表LSTM层数(包括：公共层和模型.py的专用LSTM）
    #     'multi_tasks': True,
    #     'monitor': 'T', # 多任务指定某任务的监控，或者None，默认监控总val_loss和val_mae
    #     'output_config': {
    #         'T': {'type': 'regression',  # 单变量回归
    #               'loss': 'mse',  # 主损失函数
    #               'metrics': ['mae'],  # 额外指标：平均绝对误差
    #               'loss_weights': 0.6,
    #               'units': 1,  # 每个时间步预测n个特征
    #               },
    #
    #         'rh': {'type': 'regression',
    #                'loss': 'mse',
    #                'metrics': ['mae'],
    #                'loss_weights': 0.4,
    #                'units': 1}},
    #
    #     # 首次训练配置：continue_from：None / 注意余弦配置
    #     'continue_from':None,
    #      # '/Users/shibo/AL/NeuralNetwork/saved_model/multi_lstm2*_20260305_233439/tf_checkpoints_stage0',
    #
    #     # 使用main.py继续训练（None)
    #     # 直接从continue_training.py加载训练完的最佳模型(带epoch的path)
    #     'final_best_model': None
    #      # '/Users/shibo/AL/NeuralNetwork/saved_model/multi_lstm2*_20260305_235745/tf_checkpoints_stage0/epoch_22'
    # }}

    data = {'train_data': features_temp_train, 'val_data': features_temp_val}  # 训练要求验证集

    # 准备指标计算数据（回测用，预测不用） 采样后的原数据
    valid_df_test, _ = preprocessor._get_step(3, 0).process(df_test)  # 处理连续性测试集的采样步骤（第4个class,第0个实例) 采样

    features_temp_test.to_csv(
        '/Users/shibo/AL/NeuralNetwork/temperature_forecasting/data/intermediate/analyze/testsets.csv')

    # 并行训练和预测

    def train_single_config(config, X, y, preprocessor,
                            original_data_no_scaled=None, original_data_scaled=None,
                            ):
        """
        单个模型的训练和预测流程
        Args:
            config: 模型配置
            X: 训练数据（字典-训练数据和验证数据）
            y: 标签（可能为None）
            preprocessor: 预处理器对象,提取逆转换pipeline步骤
            save_dir: 保存目录，如果为None则不保存
            original_data_no_scaled : 测试集计算指标 的原数（mape:时间列采样/连续性处理）
            original_data_scaled: 测试数据，计算指标 的原数据（mae,mse:时间列采样/连续性处理/标准化）

        Returns:
            dict: 包含模型、预测结果和postprocessor
        """
        model_name = config.get('model_type', 'unknown')
        time_column = config.get('time_column')

        try:
            logger.info(f"开始训练模型: {model_name}")

            # 1.创建模型实例
            estimator = TimeSeriesEstimator(config)

            # 2.训练（加载最优检查点）
            X_copy = copy.deepcopy(X)
            estimator.fit(X_copy, y=None)  # 包括训练集和验证集，一起用于模型训练,注意：以字典方式传递

            # 3.创建后处理器并捕获pipeline状态
            postprocessor = TimeSeriesPostProcessor(
                {'model_name': model_name,
                 'preprocessor': preprocessor,
                 'save_dir': '/Users/shibo/AL/NeuralNetwork/saved_pipeline_states',
                 'task_names': list(config.get('output_config').keys()),
                 'output_width': config.get('output_width', 1),
                 'time_col_name': config.get('time_column', 'Date Time')
                 }
            )

            # 捕获preprocessor状态
            postprocessor.capture_and_save_pipeline_state()

            # 4. 预测
            features_temp_test_data_copy = copy.deepcopy(original_data_scaled)
            raw_predictions = estimator.predict(features_temp_test_data_copy)  # 测试数据
            logger.info(
                f"测试集生成 {len(raw_predictions)} 个预测结果，每个结果代表一个预测label，形状：shape:{raw_predictions[list(config.get('output_config').keys())[0]].shape}")

            # 5. 逆标准化/编码（使用后处理器）预测数据 + 原始数据（用于训练的数值列里面的2分类列，未标准化，需要排除掉）
            inverse_predictions = postprocessor.custom_inverse_transform(
                raw_predictions=raw_predictions,
                use_saved=False,  # 是否使用【内存】中的preprocessor
                task_config=config.get('output_config'),
                output_width=config.get('output_width'),
                pipeline_name='pipeline_6',
                scale_step_names=['engineer_2', 'engineer_3'],
                transform_step_name='engineer_1')

            # 6. 添加时间戳
            final_pred_results, inversed_predictions_dict = postprocessor.add_timestamps(
                mode='train',
                predictions=inverse_predictions,
                historical_timestamps=features_temp_test_data_copy[time_column],
                input_width=config.get('input_width', 6),
                output_width=config.get('output_width', 5),
                freq='h',
                shift=config.get('shift', 0),
                save_path='/Users/shibo/AL/NeuralNetwork/temperature_forecasting/data/final/all_predictions.csv'
            )

            logger.info(f"模型 {model_name} 训练成功")
            logger.info(f"最终预测结果:{final_pred_results}")
            logger.info(f"最终预测结果形状: {final_pred_results.shape}")

            # 7. mape 业务 / mse mae 技术
            # 逐步step
            step_mape = postprocessor.calculate_mape(
                pred_data=final_pred_results,
                original_data=original_data_no_scaled.copy(),
            )
            logger.debug(f"预测结果和原数据合并表：{step_mape.tail(10)}")

            # 逐时间点timepoint / 各级别 mape mse mae
            # 模型解释mae(模型拟合程度)
            mae_mse_dict, mae_dict = MetricsCalculator.mae_mse_calculator(predictions=raw_predictions,  # 注意是标准化数据
                                                                          actual_data=features_temp_test_data_copy.copy(),
                                                                          # 标准化后的原数据（模型计算mae方式）
                                                                          input_width=config.get('input_width'),
                                                                          shift=config.get('shift'), level='o',
                                                                          time_column=config.get('time_column',
                                                                                                 'Date Time'))
            # 业务解释smape
            smape_dict = MetricsCalculator.smape_calculator(predictions=inversed_predictions_dict,  # 逆标准化数据
                                                            actual_data=original_data_no_scaled.copy(),  # 原数据
                                                            input_width=config.get('input_width'),
                                                            shift=config.get('shift'), level='o',
                                                            # o:original 原始数据时间粒度（单时间点级别）/‘D’  日级别 ‘M’月级别
                                                            time_column=config.get('time_column', 'Date Time'))

            MetricsCalculator.download_data(result_mae_mse=mae_mse_dict, result_mape=smape_dict, level_mae_mse='o',
                                            level_mape='o')
            # 业务解释mae
            MetricsCalculator.mae_mse_calculator(predictions=inversed_predictions_dict,
                                                 actual_data=original_data_no_scaled.copy(),
                                                 # 标准化后的原数据（模型计算mae方式）
                                                 input_width=config.get('input_width'),
                                                 shift=config.get('shift'), level='operational',
                                                 time_column=config.get('time_column',
                                                                        'Date Time'))

            # 8. 保存状态
            predict_window_, predict_window_config = estimator._forecast_window_generator()
            best_checkpoint = estimator.best_checkpoint  # 'saved_model/multi_lstm1_20260104_143043/tf_checkpoints/model_epoch_2/'

            deployment = DeploymentManager(
                model_name=config['model_type'],
                bestpoint=best_checkpoint,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                model_config=config,
                window_config=predict_window_config,
                window_generator=predict_window_,
                mae_dict=mae_dict
            )
            deployment.save('./temperature_forecasting/deployment_package')

            return {
                'model_name': model_name,
                'model': estimator,
                'postprocessor': postprocessor,
                'predictions': final_pred_results,
                'raw_predictions': raw_predictions,
                'config': config,
                'success': True
            }


        except Exception as e:
            logger.error(f"模型 {model_name} 训练失败: {str(e)}", exc_info=True)
            return {
                'model_name': model_name,
                'error': str(e),
                'success': False,
                'config': config
            }

    configs = [single_lstm_model_config1]  # multi_lstm_model_config2

    failed_configs = []
    trained_models = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(train_single_config, config, X=data, y=None, preprocessor=preprocessor,
                                   original_data_scaled=features_temp_test,  # 预处理后的测试集数据（时间列处理，采样，转换、标准化）
                                   original_data_no_scaled=valid_df_test,  # 预处理前的测试集数据（时间列处理，采样）
                                   )
                   for config in configs]

        for future, config in zip(futures, configs):
            try:
                result = future.result()
                trained_models.append(result)
                print("一个模型训练成功")
            except Exception as e:
                print(f"模型{config.get('model_type')}训练失败:{str(e)}")
                failed_configs.append(config)

    print(f"完成: {len(trained_models)} 个成功, {len(failed_configs)} 个失败")

    return trained_models, failed_configs


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('Agg')
    main()

# 并行的模型配置
# multi_cnn_model_config = {**base_model_config, **{
#     'model_type': 'cnn',
#     'branch_filters': [[32, 32], [64, 64]],
#     'branch_kernels': [[2, 3], [2, 3]],
#     'branch_dilation_rate': [[1, 1], [1, 1]],
#     'activation': 'relu',
#     'learning_rate': 0.001,
#     'epochs': 20,
#     'verbose': 2
# }}
