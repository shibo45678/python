import os
import argparse
import sys

# 添加项目根目录到路径
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.data_preparation import DataLoader
from predictor import TrainedModelPredictor

import logging
logger = logging.getLogger(__name__)



def run_prediction(input_data,
                   deployment_path,
                   model_name:str,
                   stage_num:int):

    if not os.path.exists(deployment_path):
        logger.debug(f"❌ 部署包不存在: {deployment_path}")
        return

    try:
        predictor = TrainedModelPredictor(deployment_path,model_name,stage_num)
        predictor.load()
        logger.debug("✅ 部署包加载成功")

        predictions,save_path = predictor.forecast(input_data, labels=None)
        logger.debug(f"✅ 预测成功!{save_path}")

        return predictions,save_path

    except Exception as e:
        logger.debug(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()


def run_with_real_data(input_file,deployment_path,model_name,stage_num):
    """使用真实数据测试"""
    logger.debug(f"=== 使用真实数据测试: {input_file} ===")
    import re
    pattern = r'(\w+)/(\w+\.\w+)'
    matches = re.finditer(pattern, input_file)

    for match in matches:
        input = match.group(2)

    try:
        loader = DataLoader(
            input_files=[input],
            pattern="new_*.csv",
            data_dir='./data/raw'
        )
        new_features = loader.learn_process()
        logger.debug(f"加载数据: {len(new_features)} 行")

        run_prediction(input_data=new_features,
                       deployment_path =deployment_path ,
                       model_name =model_name, # model_type
                       stage_num=stage_num)

    except Exception as e:
        logger.debug(f"❌ 加载数据失败: {e}")



def main():
    '''测试部署包'''
    parser = argparse.ArgumentParser(description='模型预测')
    parser.add_argument('--input', default='./data/raw/latest_data.csv',
                        help='输入数据路径')
    parser.add_argument('--deployment_path',
                        default='/Users/shibo/AL/NeuralNetwork/temperature_forecasting/deployment_package/',
                        help='部署包所在目录')
    parser.add_argument('--model_name', default='single_lstm1',
                        help='模型名称')
    parser.add_argument('--stage_num', type=int, default=0,
                        help='阶段编号')
    args = parser.parse_args()



    if os.path.exists(args.input):
        run_with_real_data(input_file=args.input,
                           deployment_path=args.deployment_path,
                           model_name=args.model_name,
                           stage_num=args.stage_num)
    else:
        logger.debug(f"文件不存在：{args.input}")
        logger.debug("使用 --input 参数进行示例测试")


if __name__ == '__main__':
    main()

