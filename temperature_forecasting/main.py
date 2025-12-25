# 优化方向：
# 追加参数检测 window / handle_extre_numeric
# process_categorical_cols 支持排除列
# 时间列处理中新生成列 独立放入feature_generator
# 每种类型里面 生成新的特征 模块点 小标题系统点（风矢量是数值列、process将encoding onehot 纳入、feature generator 可以独立、时间列（单）处理和生成分离
# 模型output_config是单列配置，优化成同一类配置可以一起跑。什么都不写默认回归？分类？
# 缺失值填充 增加算法填充等
# missing列logger 从提示改为报错 尽早暴露/结果无意义/
# 统一下划线/标准化列有不存在需要提前验证
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from sklearn.utils.validation import check_is_fitted
from utils.tensorflow_config import TensorFlowConfig
from models.NeuralNetwork import TimeSeriesEstimator, TimeSeriesPostProcessor
from pipelines.preprocess_pipeline import CompletePreprocessor
from data.data_preprocessing import TimeSeriesSplitter
from data.data_preparation import (DataLoader, DescribeData, RemoveDuplicates, DeleteUselessCols, ProblemColumnsFixed,
                                   SpecialColumnsFixed, CheckExtreFeatures,
                                   NumericOutlierProcessor, detect_, handle_,
                                   CategoricalOutlierProcessor, NumericMissingValueHandler,
                                   CategoricalMissingValueHandler,
                                   ColumnsTypeIdentify,
                                   ConvertCategoricalColumns,
                                   ConvertNumericColumns, ProcessTimeseriesColumns)
from data.data_preprocessing import SimpleTimeSampler
from data.feature_engineering import (GenerationFromNumeric,GenerationFromTimeseries,  BasedOnCorrSelector,
                                      UnifiedFeatureScaler, CategoricalEncoding)
from data.exploration import VisualizationForNeural
import logging.config
from logging_config import LOGGING_CONFIG
import logging

logger = logging.getLogger(__name__)


def main():
    logging.config.dictConfig(LOGGING_CONFIG)
    TensorFlowConfig.setup_environment()
    '''
    1. 严格按照数据的【处理顺序】使用‘class’，并标记'len_change'(这里将改变数据长度的步骤，手动处理）
    2. 手动处理的类:是无法放进pipeline的类，不会继承BaseEstimator和TransfromerMixin。并且使用learn_process处理。
    3. 一次训练直接 predict（keras格式），
       隔日预测 predict_with_best_checkpoint(new_data)（keras格式） ，
       专用部署 save + load(格式Saved Model)
    '''
    check_outliers_config = {'method': 'iqr', 'threshold': 1.5}
    download_duplicates_config = {
        'enabled': True,
        'path': '~/Python/NeuralNetwork/temperature_forecasting/data/intermediate',
        'filename': 'duplicate_rows.csv'}
    download_outliers_details_config = {
        'enabled': True,
        'path': '~/Python/NeuralNetwork/temperature_forecasting/data/intermediate',
        'filename': 'outliers.csv'}

    numeric_outliers_config = {'detect_and_handle_config': [
        {'zscore': {'columns': ['T', 'Tpot', 'Tdew'], 'handle_method': ['clip', 'clip', 'clip'], 'threshold': 3}},
        # 气温常接近正态分布，Z-score效果好
        {'iqr': {'columns': ['p', 'VPmax', 'VPact', 'VPdef'], 'handle_method': ['clip', 'clip', 'clip', 'clip'],
                 'threshold': 1.5, 'handle_function': partial(handle_)}},  # 气压有明确的物理范围，IQR对中等离群值敏感
        {'robust':
             {'columns': ['rh', 'sh', 'H2OC'], 'handle_method': ['clip', 'clip', 'clip'],
              'quantile_range': (5, 95), }},
        # 分位数检测对分布偏斜

        {'isolationforest': {'columns': ['rho', 'wd'], 'handle_method': ['clip', 'clip'], 'contamination': 0.025,
                             'random_state': 42}},
        # 对复杂分布效果好
        {'custom': {'columns': ['wv', 'max. wv'], 'handle_method': ['custom', 'custom'],
                    'detect_function': partial(detect_, threshold=0),
                    'handle_function': partial(handle_)}}],
        'generate_outlier_indicator': ['T', 'rh']
    }

    categorical_outliers_config = {
        'rare_threshold': 0.01,
        'similarity_threshold': 0.8,
        'auto_correct_typos': True}

    numeric_missing_config = {
        'spec_fill': [
            {'constant': {'columns': ['wv', 'max. wv'], 'fill_value': [0, 0]}},
            {'mode': {'columns': []}},
            {'ffill': {'columns': ['p']}},  # 气压变化相对连续稳定，短期内有持续性
            {'bfill': {'columns': ['rh']}}  # 湿度变化相对缓慢，受天气系统影响有持续性
        ],
        'skip_fill': [],
        'smart_fill_remain': True,
        'important_columns': ['T', 'rh'],
    }
    # 新特征生成后
    scaling_config = {
        'transformers': [
            {'standard': {
                'columns': ['p', 'Tpot', 'Tdew', 'wv_x', 'wv_y', 'max. wv_x', 'max. wv_y']}},
            {'minmax': {'columns': ['rh', 'VPmax', 'Vpact', 'VPdef', 'sh', 'H2OC', 'rho'], 'feature_range': (0, 1)}},
            # 相同方法，相同其他参数配置，在columns列表填写
            {'minmax': {'columns': ['T'], 'feature_range': (-1, 1)}},  # 相同方法，但是其他参数配置与前一配置不同，允许在下一行填写
            {'robust': {'columns': [], 'quantile_range': (10, 90)}}
        ],
        'skip_scale': ['is_night', 'Day_sin', 'Day_cos', 'Year_sin', 'Year_cos', 'Month_sin', 'Month_cos','Season_sin','Season_cos']
        # 跳过二分类列(数值型）/ 异常值标记列自动skip
    }

    # 先初始化 再延迟计算（lazy evaluation）
    preparation_configs = [
        {'obj_list': [DescribeData(log_level="DEBUG"), DeleteUselessCols()], 'len_change': False},
        {'obj_list': [RemoveDuplicates(download_config=download_duplicates_config),], 'len_change': True},
        {'obj_list': [ColumnsTypeIdentify(),
                      ConvertCategoricalColumns(categorical_columns=[]),
                      ConvertNumericColumns(preserve_object_integer_types=True, exclude_cols=['Date Time']),
                      ProblemColumnsFixed(problem_columns=['wv']), SpecialColumnsFixed(problem_columns=['T']),  # wv 一样
                      CheckExtreFeatures(method_config=check_outliers_config,
                                         download_config=download_outliers_details_config),
                      NumericOutlierProcessor(method_config=numeric_outliers_config),  # iqr / 业务初筛 --> 异常值初筛
                      CategoricalOutlierProcessor(method_config=categorical_outliers_config, strategy='consolidate'),
                      NumericMissingValueHandler(method_config=numeric_missing_config),  # 填充缺失值(时间缺失 暂不填充。等采样完）
                      CategoricalMissingValueHandler(method_config=None, pass_through=True),  # 后续隔离森林精细去异常
                      ],
         'len_change': False},

        {'obj_list': [SimpleTimeSampler(time_column='Date Time', freq_hours=1, minute=0, second=0)], 'len_change': True},

        {'obj_list': [GenerationFromNumeric(dir_cols=['wd'], var_cols=['wv', 'max. wv'], plot=False),
                      GenerationFromTimeseries(time_column='Date Time',plot=False),
                      BasedOnCorrSelector(pass_through=True),
                      UnifiedFeatureScaler(method_config=scaling_config, algorithm='lstm'),  # 自动根据数据分布及算法类型进行推荐标准化
                      CategoricalEncoding(handle_unknown='ignore', unknown_token='__UNKNOWN__'),
                      VisualizationForNeural(pass_through=True),
                      ], 'len_change': False},
    ]

    # 1. 加载数据
    loader = DataLoader(input_files=['data_climate.csv'], pattern="new_*.csv", data_dir='data/raw')
    raw_data = loader.learn_process()

    # 2. 数据集分割
    splitter = TimeSeriesSplitter(train_size=0.8, val_size=0.15, test_size=0.05, shuffle=False)
    df_train, df_val, df_test = splitter.learn_process(raw_data)
    logger.info(f"训练集数：{len(df_train)}，验证集数:{len(df_val)}，测试集数：{len(df_test)}。")

    # 3. 检查预测数据集时间列的连续性（时间列智能检测并转换 + 时间序列缺失情况） Date Time
    time_detector = ProcessTimeseriesColumns(interactive=False, create_extract_continuous=True)
    valid_df_test,_ = time_detector.learn_process(df_test,y=None)
    time_col = time_detector.valid_time_column_

    # 4. 数据预处理(生成训练、验证、预测数据）
    preprocessor = CompletePreprocessor(preparation_configs)
    features_temp_train, _ = preprocessor.train(features=df_train, labels=None)

    # 立即检查状态
    # logger.debug("=== 训练后立即检查 ===")
    # for name, pipeline in preprocessor.pipelines_.items():
    #     logger.debug(f"{name}:")
    #     try:
    #         check_is_fitted(pipeline)
    #         logger.debug("  ✓ Pipeline 整体已拟合")
    #     except Exception as e:
    #         logger.debug(f"  ✗ Pipeline 整体未拟合: {e}")
    #
    #     # 检查每个步骤
    #     for step_name, transformer in pipeline.steps:
    #         try:
    #             check_is_fitted(transformer)
    #             logger.debug(f"    ✓ {step_name} 已拟合")
    #         except Exception as e:
    #             logger.debug(f"    ✗ {step_name} 未拟合: {e}")

    features_temp_val, _ = preprocessor.transform_predict(features=df_val, labels=None)
    features_temp_test, _ = preprocessor.transform_predict(features=valid_df_test, labels=None)

    num_cols = preprocessor.get_specific_attribute(4, 'engineer_3', 'numeric_columns_')  # 取第5个class的第4步的属性
    cat_cols = preprocessor.get_specific_attribute(4, 'engineer_4', 'categorical_columns_')

    # 4. 并行模型训练、评估
    # single_base_model_config = {'numeric_columns': num_cols,
    #                             'categorical_columns': cat_cols,
    #                             'time_column': time_col,
    #                             'input_width': 6,
    #                             'output_width': 5,
    #                             'shift': 4,
    #                             'output_config': {
    #                                 'T': {'type': 'regression',
    #                                       'loss': 'mse',
    #                                       'metrics': ['mae'],
    #                                       'loss_weights': 1,
    #                                       'units': 1,
    #                                       }},
    #                             'multi_tasks':False}
    #
    # single_lstm_model_config1 = {**single_base_model_config, **{
    #     'model_type': 'single_lstm1',
    #     'learning_rate': 0.00035,
    #     'units': [192],  # len控制lstm的层数
    #     'return_sequences': [False],
    #     'epochs': 30,
    #     'verbose': 2
    # }}

    multi_base_model_config = {'numeric_columns': num_cols,
                               'categorical_columns': cat_cols,
                               'time_column': time_col,
                               'input_width': 6,
                               'output_width': 5,
                               'shift': 24,

                               'output_config': {
                                   'T': {'type': 'regression',  # 单变量回归
                                         'loss': 'mse',  # 主损失函数
                                         'metrics': ['mae'],  # 额外指标：平均绝对误差
                                         'loss_weights': 0.95,
                                         'units': 1,  # 每个时间步预测n个特征
                                         },

                                   'rh': {'type': 'regression',
                                          'loss': 'mse',
                                          'metrics': ['mae'],
                                          'loss_weights': 0.22,
                                          'units': 1,
                                          }
                               },
                               'multi_tasks': True,
                               }

    multi_lstm_model_config1 = {**multi_base_model_config, **{
        'model_type': 'multi_lstm1',
        'learning_rate': 0.00035,
        'units': [192],  # len控制lstm的层数
        'return_sequences': [False],
        'epochs': 3,
        'verbose': 2
    }}

    data = {'train_datasets': features_temp_train, 'val_datasets': features_temp_val}  # 训练要求验证集

    # 并行训练和预测
    def train_single_config(config, X, y, preprocessor, new_data,
                            save_dir=None):
        """
        单个模型的训练和预测流程

        Args:
            config: 模型配置
            X: 训练数据（字典-训练数据和验证数据）
            y: 标签（可能为None）
            preprocessor: 预处理器对象,提取逆转换pipeline步骤
            historical_timestamps: 历史时间戳，用于预测结果展示
            new_data: 测试 / 新数据（用于一次预测）
            save_dir: 保存目录，如果为None则不保存
        Returns:
            dict: 包含模型、预测结果和postprocessor
        """
        model_name = config.get('model_type', 'unknown')
        time_column = config.get('time_column')

        try:
            logger.info(f"开始训练模型: {model_name}")

            # 1.创建模型实例
            model = TimeSeriesEstimator(config)

            # 2.训练（加载最优检查点）
            X_copy = copy.deepcopy(X)
            model.fit(X_copy, y=None)  # 包括训练集和验证集，一起用于模型训练,注意：以字典方式传递

            # 3.创建后处理器并捕获pipeline状态
            postprocessor = TimeSeriesPostProcessor(
                {'model_name': model_name,
                 'preprocessor': preprocessor,
                 'save_dir': save_dir,
                 'task_names': list(config.get('output_config').keys()),
                 'output_width': config.get('output_width', 1),
                 'time_col_name':config.get('time_column','Time')
                 }
            )

            # 捕获preprocessor状态
            postprocessor.capture_and_save_pipeline_state()

            # 4. 预测
            features_temp_data_copy = copy.deepcopy(new_data)
            raw_predictions = model.predict(features_temp_data_copy)  # 测试数据
            logger.info(f"测试集生成 {len(raw_predictions)} 个预测结果，每个结果代表一个预测label，形状：shape:{raw_predictions[0].shape}")

            # 5. 逆转换（使用后处理器）
            inverse_predictions = postprocessor.custom_inverse_transform(
                raw_predictions=raw_predictions,
                use_saved=False,  # 使用【内存】中的preprocessor
                pipeline_name='pipeline_4',
                step_names=['engineer_3', 'engineer_4'],
                target_columns=list(config.get('output_config').keys()))

            # 6. 添加时间戳
            final_results = postprocessor.add_timestamps(
                predictions=inverse_predictions,
                historical_timestamps=features_temp_data_copy[time_column],
                input_width = config.get('input_width',6),
                output_width=config.get('output_width', 5),
                freq='h',
                shift=config.get('shift', 0),
            )
            logger.info(f"模型 {model_name} 训练成功")
            logger.info(f"最终预测结果:{final_results}")
            logger.info(f"最终预测结果形状: {final_results.shape}")

            # mape（验证集和测试集都需要）


            # 7. 保存状态
            if save_dir:
                model_save_dir = os.path.join(save_dir, model_name)
                os.makedirs(model_save_dir, exist_ok=True)

                # 保存模型（调整）
                # 保存后处理器状态
                postprocessor.save_state(model_save_dir)
                # 保存预测结果
                results_file = os.path.join(model_save_dir, 'predictions.csv')
                final_results.to_csv(results_file, index=False)
                logger.info(f"模型状态已保存到：{model_save_dir}")

            return {
                'model_name': model_name,
                'model': model,
                'postprocessor': postprocessor,
                'predictions': final_results,
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

    # parallel_train_all_models

    configs = [multi_lstm_model_config1]  # multi_cnn_model_config

    failed_configs = []
    trained_models = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(train_single_config, config, X=data, y=None, preprocessor=preprocessor,
                                   new_data=features_temp_test,
                                   save_dir='/Users/shibo/Python/NeuralNetwork/saved_model_state')
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

# save的节点 是否是最佳模型
# 多变量输出 如何协调权重，以及梯度剪裁

# 并行的模型配置
# multi_lstm_model_config2 = {**base_model_config, **{
#     'model_type': 'multi_lstm2',
#     'learning_rate': 0.001,
#     'units': [64, 32],  # 逐步压缩特征
#     'return_sequences': [True, False],  # 上一轮的输出做本轮输入input + 上一轮输出
#     'epochs': 50,
#     'verbose': 2
# }}

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


# 直接加载训练模型并预测
# predictor = TrainedModelPredictor("/Users/shibo/Python/NeuralNetwork/saved_model/multi_lstm1_20251219_100333/tf_checkpoints/model_epoch_2/")
# predictions = predictor.predict(new_cleaned_data)
# 预测最后一步的处理