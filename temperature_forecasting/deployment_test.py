import sys
import os
import argparse
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.data_preparation import DataLoader
from predictor import TrainedModelPredictor





def run_prediction(input_data):
    deployment_path = '/Users/shibo/Python/NeuralNetwork/deployment_package'
    if not os.path.exists(deployment_path):
        print(f"❌ 部署包不存在: {deployment_path}")
        return

    try:
        predictor = TrainedModelPredictor(deployment_path)
        predictor.load()
        print("✅ 部署包加载成功")

        predictions = predictor.forecast(input_data, labels=None)

        print(f"✅ 预测成功!")

        for task_name, pred in predictions.items():
            task_name = pd.DataFrame(pred, columns=[f'{task_name}_{j}' for j in range(pred.shape[1])])
            task_name.to_csv(
                f'/Users/shibo/Python/NeuralNetwork/temperature_forecasting/data/intermediate/test_{task_name}_predictions.csv',
                index=False)

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()


def run_with_real_data(input_file):
    """使用真实数据测试"""
    print(f"=== 使用真实数据测试: {input_file} ===")
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
        print(f"加载数据: {len(new_features)} 行")

        run_prediction(new_features)

    except Exception as e:
        print(f"❌ 加载数据失败: {e}")



def main():
    parser = argparse.ArgumentParser(description='测试部署包')
    parser.add_argument('--input', help='输入数据文件', default='./data/raw/latest_data.csv')
    parser.add_argument('--output', help='输出文件', default='predictions.csv')
    parser.add_argument('--sample', action='store_true', help='使用示例数据')
    args = parser.parse_args()


    if os.path.exists(args.input):
        run_with_real_data(args.input)
    else:
        print(f"文件不存在：{args.input}")
        print("使用--sample 参数进行示例测试")


if __name__ == '__main__':
    main()
