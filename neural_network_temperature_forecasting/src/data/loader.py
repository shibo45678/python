from pathlib import Path
import codecs
import os
import csv
import pandas as pd
from typing import Dict
import glob
import numpy as np


class EncodingHander:
    def __init__(self, input_files: list):
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.dir_path = PROJECT_ROOT / "data"
        self.input_files = input_files

    def handleEncoding(self):
        # 1.指定文件所在目录 - 根据源文件名进行命名"new_"
        # 2.创建空csv文件（拼接路径 os.path.join() - 写入表头）
        newfile_names = []
        for i in self.input_files:
            newfile_name = "new_" + i
            newfile_names.append(newfile_name)

        for file in newfile_names:
            new_file_path = os.path.join(self.dir_path, file)
            # 创建csv写入器
            with open(new_file_path, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)  # writer.writerow(["",""]) # 写入表头

        """按照确定的 encoding 读取旧文件内容，另存为utf-8编码内容的新文件"""
        for i in range(len(self.input_files)):
            original_file = os.path.join(self.dir_path, self.input_files[i])
            new_file = os.path.join(self.dir_path, newfile_names[i])

            f = open(original_file, "rb+")
            content = f.read()  # 读取文件内容，content为bytes类型，而非string类型
            source_encoding = "utf-8"  # 初始化source_encoding

            try:
                # 尝试以不同的编码解码内容
                for encoding in ["utf-8", "gbk", "gb2313", "gb18030", "big5", "cp936"]:
                    try:
                        decode_content = content.decode(encoding)
                        source_encoding = encoding
                        break  # 如果找到匹配的编码，就跳出循环
                    except UnicodeDecodeError:
                        pass  # 如果解码失败，继续尝试其他编码
                else:  # 如果循环结束还没有找到匹配的编码
                    print("无法确定原始编码")

            except Exception as e:
                print(f"发生错误：{e}")
            finally:
                f.close()  # 确保文件总是关闭的

            # 编码：读取-存取
            block_size = 4096
            with codecs.open(original_file, "r", source_encoding) as f:
                with codecs.open(new_file, "w", "utf-8") as f2:
                    while True:
                        content = f.read(block_size)
                        if not content:
                            break
                        f2.write(content)

class DataLoader:
    def __init__(self):
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.dir_path = PROJECT_ROOT / "data"
        self.merged_df =pd.DataFrame()

    def load_all_data(self,pattern)->pd.DataFrame:
        """多个新文件并合并"""
        full_pattern = os.path.join(self.dir_path,pattern)
        all_files=glob.glob(full_pattern) # 获取解析后的文件

        print(f"搜索模式：{full_pattern}")
        print(f"找到的文件：{all_files}")

        data_frames =[]
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                data_frames.append(df)
                print(f"成功读取: {file_path}, 形状: {df.shape}, dtype:{df.dtypes}")
            except Exception as e:
                print(f"读取文件失败{file_path}:{str(e)}")

        if data_frames:
            self.merged_df = pd.concat(data_frames,ignore_index=True)

            # 保存合并后的文件
            output_file = "merged_data.csv"
            output_path = os.path.join(self.dir_path, output_file)
            self.merged_df.to_csv(output_path, index=False)
            print(f"合并文件将保存到: {output_path}")

            return self.merged_df
        else:
            raise ValueError(f"没有找到匹配的文件或所有文件读取失败")


class FixProblemColumns: # 可以不重新编码，直接读试试
    def __init__(self,
                 dataloader:DataLoader,
                 problem_columns:list=None):
        self.dataloader = dataloader
        self.problem_columns = problem_columns
        self.original_df = self.dataloader.merged_df
        self.fixed_df = self.original_df.copy()


    def problem_columns_fixed(self):
        """修复常见特殊问题-正则修复"""
        need_fix_df = self.original_df.copy()
        if self.problem_columns is None:
            raise ValueError(f"待修复列列表为空")

        # 检查问题列
        for col in self.problem_columns:
            if col in need_fix_df.columns:
                sample_value = need_fix_df[col].iloc[0] if len(need_fix_df[col]) > 0 else None
                print(f"问题列第一个元素：{sample_value}")
                need_fix_df[col] = need_fix_df[col].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
                self.fixed_df[col] = pd.to_numeric(need_fix_df[col], errors='coerce')
                print(f"已转文本，正则清洗，转回数值")
            else:
                print(f"问题列{col}不在 merged_df的列中")
                continue

        return self

    def special_columns_fixed(self):
        """特殊列问题问题"""
        print("=== 深入诊断问题列 ===")
        need_fix_df = self.original_df.copy()

        for col in self.problem_columns:
            # 检查Series内部结构
            print(f"Series 类型:{type(need_fix_df[col])}")
            print(f"Series dtype:{need_fix_df[col].dtype}")
            print(f"Series形状:{need_fix_df[col].shape}")

            # 先提取再处理
            series = need_fix_df[col].copy()

            # 一、检查第一个非空值的实际类型,是否是DataFrame（尝试修复）
            first_check = series.dropna(inplace=False)
            first_non_null = first_check.iloc[0] if not first_check.empty else None
            print(f"问题列{col}的第一个元素的dtype:{type(first_non_null)}")

            if isinstance(first_non_null,pd.DataFrame):
                print(f"确认{col}列包含DataFrame对象，第一个元素形状：{first_non_null.shape}")

                if hasattr(first_non_null,'iloc'):
                    print("第一个元素有iloc方法")
                    try:
                        inner_value = first_non_null.iloc[0,0] if first_non_null.shape[1] >0 else first_non_null.iloc[0]
                        print(f"内部值：{inner_value}（类型：{type(inner_value)})")
                    except:
                        print("无法访问内部值")

                # 提取每个DataFrame的第一个值
                extracted_values = []
                for i, inner_value in enumerate(series):
                    if isinstance(inner_value, pd.DataFrame) and not inner_value.empty:
                        # 提取第一个单元格的值
                        extracted_values.append(inner_value.iloc[0, 0] if inner_value.shape[1] > 0 else np.nan)
                    else:
                        extracted_values.append(inner_value)

                # 创建新的Series
                series_fixed = pd.Series(extracted_values, index=series.index, name=col)

                # 替换原列
                self.fixed_df[col] =series_fixed
                print(f"修复后的{col}列类型: {type(self.fixed_df[col].iloc[0])}")

            # 二、检查整列是否有DataFrame（未尝试修复）
            elif any(isinstance(inner_value, pd.DataFrame) for inner_value in series if pd.notna(inner_value)):
                   print(f"{col}列中有其他单元格包含DataFrame对象,暂未修复")

            # 三、检查无果填充原值
            else:
                print(f"{col}列不包含DataFrame对象，无需修复")

            # 四、检查Series是否在某些操作下表现出DataFrame行为
            print("\n=== 行为测试 ===")
            # 测试1：尝试转置
            try:
                transposed = series.transpose()
                print(f"转置结果类型：{type(transposed)}")
                if hasattr(transposed,'shape'):
                    print(f"转置形状：{transposed.shape}")
            except Exception as e:
                print(f"转置失败：{str(e)}")

            # 测试2：尝试访问列
            try:
                if hasattr(series,'columns'):
                    print(f"有columns属性：{series.columns}")
                else:
                    print("没有columns属性")
            except Exception as e:
                print(f"检查columns失败：{str(e)}")

        return self

    def get_fixed_data(self):
        return self.fixed_df.copy()



# PROJECT_ROOT = Path(__file__).parent.parent.parent
# dir_path = PROJECT_ROOT / "data" /"new_data_climate.csv"
# df = pd.read_csv(dir_path)
#
# t_val=df.pop('T').values
# t_idx=df.index
#
# new_fixed_df = pd.DataFrame(index=t_idx)
# for col in df:
#     new_fixed_df[col] = df[col].values
#
# print(new_fixed_df.columns.tolist())







