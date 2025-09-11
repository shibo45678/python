from pathlib import Path
import pandas as pd
from typing import Dict, Tuple


def load_data() -> Tuple:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    input_file1 = PROJECT_ROOT / "data" / "data_climate.csv"
    input_file2 = PROJECT_ROOT / "data" / "data_climate_detail.csv"


    data1 = pd.read_csv(input_file1)
    data2 = pd.read_csv(input_file2)

    return data1, data2


def format(df: pd.DataFrame) -> Dict:
    col_type = {}
    for col in df.columns:
        col_type[f"{col}"] = type(df.loc[0, col])
    return col_type


print("==========第一个文件==========")
print(load_data()[0].head(10))
print("==========每列的格式如下==========")
print(format(load_data()[0]))
print("==========第二个文件==========")
print(load_data()[1].head(10))
print("==========每列的格式如下==========")
print(format(load_data()[1]))
