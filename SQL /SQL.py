from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType
import os
from pyspark.sql import SparkSession
#
# # 设置使用统一的Python路径（查找你系统上的python3.9路径）
# os.environ['PYSPARK_PYTHON'] = '/Users/shibo/anaconda0412/anaconda3/bin/python'  # 或你的python3.9路径
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/shibo/anaconda0412/anaconda3/bin/python'
#
#
# spark = SparkSession.builder \
#         .appName("myapp") \
#         .master("local[*]") \
#         .config("spark.sql.shuffle.partitions", "4") \
#         .config("spark.driver.memory", "2g") \
#         .getOrCreate()
#
#
# user_data=pd.read_excel('user_data.xlsx',
#                         usecols=['销售月份','大类名称','顾客编号','销售金额'],
#                         dtype={'销售月份': str})
# threshold_data=pd.read_excel('threshold_data.xlsx',usecols=['大类名称','fixed'],
#                              dtype={'fixed':float})
#
# user_data_01 = user_data[user_data['销售月份']=='2015-01']
#
# user_stats_01 = spark.createDataFrame(user_data_01)
# dynamic_adjust = spark.createDataFrame(threshold_data)
# # 确保fixed列是double类型
# dynamic_adjust = dynamic_adjust.withColumn("fixed", F.col("fixed").cast("double"))
#
# # 合并数据(join)	pd.merge(df1, df2, on='key')	df1.join(df2, on='key')
# merged_df = user_stats_01.join(
#     dynamic_adjust.select("大类名称","fixed"),
#     on="大类名称",
#     how="left")
# # 定义窗口 + 批量计算
# window_spec = Window.partitionBy("大类名称") \
#     .orderBy("销售金额") \
#     .rangeBetween(-F.col("fixed"), F.col("fixed")) # 对每行，用该行的 fixed 值作为范围半径
#    # 列表达式是 Spark SQL 的核心操作单元。rangeBetween() 接受列表达式实现动态范围。
#
# result_df = merged_df.withColumn(
#     "similar_users_01",
#     F.count("*").over(window_spec)
# )
#
# result_df.show()

"""连接数据库 批量查询"""
'''SELECT * FROM users WHERE'''
# import pandas as pd
# import sqlite3
# def build_query(**filters):
#     query = f"SELECT * FROM users WHERE "
#     conditions = []
#     for key, value in filters.items(): # name:  age:
#         # ? 是占位符，真正的值是在执行查询时通过参数传递的，不会直接拼接到 SQL 字符串中
#         # cursor.execute(query, params)才是执行
#         conditions.append(f"{key} = ?")  # 生成 "name = ?" 和 "age = ?" 防止 SQL 注入
#     return query + " AND ".join(conditions), list(filters.values())
# def fetch_users(db_path, **kwargs):
#     query, params = build_query(**kwargs)
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     # cursor.execute("SELECT ... WHERE name = ? AND age = ?", ["Alice", 25])
#     cursor.execute(query, params)  # 安全传参
#     return cursor.fetchall()
# # 批量查询
# df = pd.read_csv("users.csv")  # 假设 CSV 有 name, age, role 等列
# queries = df.to_dict("records")  # 转换成 [{"name": "Alice", "age": 25}, ...]
#
# for params in queries:
#     results = fetch_users("example.db", **params) # 键值参数
'''cume——dist'''
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import cume_dist, col

spark = SparkSession.builder.appName("CumeDistExample").getOrCreate()

# 示例数据
data = [(1, 100), (2, 150), (3, 200), (4, 250), (5, 300), (6, 350), (7, 400)]
df = spark.createDataFrame(data, ["order_id", "amount"])
# 正确的cume_dist用法
window_spec = Window.orderBy("amount")
result_df = df.select(
    col("order_id"),
    col("amount"),
    cume_dist().over(window_spec).alias("amount_distribution")
)
result_df.show()

# 输出结果会与SQL版本一致

