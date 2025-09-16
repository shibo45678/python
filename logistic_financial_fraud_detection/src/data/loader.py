from pathlib import Path
import pandas as pd


def load_data():
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    input_file1 = PROJECT_ROOT / "data" / "data_creditcard.csv"

    data = pd.read_csv(input_file1, encoding='gbk')

    return data


print(load_data().head(5))
