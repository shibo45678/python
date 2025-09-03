import pandas as pd
import numpy as np
import matplotlib as plt  # 新增导入

# 清除matplotlib缓存（可选）
plt.rcParams.update(plt.rcParamsDefault)
# from pyspark.sql import SparkSession
# import pyspark.sql.functions as F

# spark = SparkSession.builder \
#         .appName("myapp") \
#         .master("local[*]") \
#         .config("spark.sql.shuffle.partitions", "4") \
#         .config("spark.driver.memory", "2g")\
#         .config("spark.jars.packages", "com.crealytics:spark-excel_2.12:3.5.0_0.20.3") \
#         .getOrCreate()


# from dateutil.relativedelta import relativedelta
# # import pandas as pd
# # l = pd.read_excel('/Users/shibo/pythonProject1/可视化/operational_data.xlsx', sheet_name='Sheet1')
#
# # l = pd.read_excel('/Users/shibo/pythonProject1/可视化/1.xlsx')
# #
# # j = 1  # 全局 j，所有行共享
# # for i, row in l.iterrows():
# #     if j < 10:
# #         l.loc[i, '业务发生结算月份'] = f"{row['业务发生结算月份']}-0{j}"
# #     else:
# #         l.loc[i, '业务发生结算月份'] = f"{row['业务发生结算月份']}-{j}"
# #
# #     j += 1
# #     if j == 31:  # 如果 j=31，重置为 1
# #         j = 1
# #
# # print(l['业务发生结算月份'])
#
# # shuffled_l = l.sample(frac=1).reset_index(drop=True)
# # cur
# # shuffled_l.to_excel('operational_data-nodate.xlsx')
#
#
# # a ='2025-06-16'
# # c = date.today()
# #
# # if c.month == pd.to_datetime(a).month :
# #     print(c)
# # else:
# # print('不相等',a,b,c,d)
# import pandas as pd
# import numpy as np
#
# # l = pd.read_excel('/Users/shibo/pythonProject1/可视化/operational_data.xlsx', sheet_name='Sheet1')
# # l1=l['业务发生结算月份']
# # o = pd.to_datetime(l1)
# # print(o)
#
#
# # end_date = date.today()
# # start_date = end_date - relativedelta(years=1)
# # date_range = pd.date_range(start_date, end_date, freq='M')
# # months = [date.strftime('%Y-%m') for date in date_range]  # 将datetime 转为 字符串
#
#
# # 对齐索引并填充缺失值
# # dict = {'a': ['2025-01-01', '2025-01-02'], }
#
# #
#
#
#
# # import pandas as pd
# #
# # d = {'month': ['2025-01', '2025-01', '2025-02'],
# #         'rev': [200, 200, 400]}
# # a = pd.DataFrame(dict)
# #
# # group1= a.groupby(['month']).sum()
# # group2 = ['2025-01','2025-02', '2025-04']
# # print(group1)
# #
# # fill_group = pd.DataFrame(columns=group1.columns)
# # for month in group2:
# #         if month not in(group1.index):
# #             # 为每个缺失的月份添加一行 0
# #             fill_group.loc[month] = 0  # 直接使用 loc 添加行
# #
# # group_filled = pd.concat([group1,fill_group]) # 中括号
# # print(group_filled)
# #
# # import pandas as pd
# #
# # data = {'month': ['2025-01', '2025-01', '2025-02'],
# #         'rev': [200, 200, 400]}
# # a = pd.DataFrame(data)
# #
# # # 按月份分组求和
# # group1 = a.groupby(['month']).sum()
# #
# # # 你想要的完整月份列表
# # group2 = ['2025-01', '2025-02', '2025-04']
#
# # 创建一个包含所有月份的DataFrame，初始值为0
# # full_index = pd.DataFrame(index=group2, columns=group1.columns).fillna(0)
# #
# # print(group1)
# # print(full_index)
# # # 用df2中的数据填充df1中的NaN，也就是说，【df1有优先权】，只有当df1中某个位置为NaN时，才会用df2中相同位置的值来填充。
# # group_filled2 = group1.combine_first(full_index)
# # print(group_filled2)
# #
# # df1 = pd.DataFrame({'A': [None, 0], 'B': [None, 4]})
# # df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
# #
# # print(df1,df2,df1.combine_first(df2))
#
#
#
# # # axis=0：操作行，结果影响行 ; axis=1：操作列，结果影响列
# # df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
# # df2 = pd.DataFrame({'B': [3, 4]}, index=[0, 1])
# # df3 = pd.concat([df1, df2], axis=1) # 左右合并（结果增加/影响 列）
# # df3
# # # axis=0：操作行，结果影响行 ; axis=1：操作列，结果影响列
# # # 数据删除
# # df4 = df3.drop('A', axis=1)  # 删除列'A' (结果影响列)
# # df4
# #
# # # axis=0：操作行，结果影响行 ; axis=1：操作列，结果影响列
# # # 统计类
# # df5 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
# # df6 = df5.sum(axis=1)  # 计算每行的总和，按行计算，（新增一列,结果影响列）
# # df6
# # # axis=0：操作行，结果影响行 ; axis=1：操作列，结果影响列
# # # 填充缺失值（fillna)
# # df2.fillna(method='ffill', axis=1)  # 用左侧列的值填充右侧NaN( 结果影响列）
# #
# # # ----------------
# # # 按行应用函数（axis=0：操作行，结果影响行 ; axis=1：操作列，结果影响列 -- 反过来）
# # df7 = df3.apply(lambda row: row['A'] + row['B'], axis=1) # 结果影响行
# # df7
#
#
# #
# # b =a.groupby(['month'])['rev'].sum()
# # c= a.groupby(['month'])['pro'].sum()
# #
# # d = (c/b).reset_index()
# # e = round(a.groupby(['month'])['pro'].sum()  / a.groupby(['month'])['rev'].sum() ,2)
# # print(e)
#
# # def a(dept):
# #     a = 20
# #     b = 30
# #     c = 50
# #     return a, b, c
# # result_dict = a('传参')
# #
# # def b(dept_1):
# #     b = 30
# #     c = 50
# #     return  b, c
# # result_a = a('传参')
# # result_b = b('传参')
# # print(result_a,result_b)
# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
# fig.update_layout(legend=dict(orientation='h'))
# fig.show()
'''sql 排序'''
# Data = {'user_id':[1, 1, 1, 1, 2, 2, 2, 3, 3],
#         'order_date':['2023-01-10', '2023-01-05', '2023-01-15', '2023-01-01',
#                   '2023-02-01', '2023-02-10', '2023-02-05', '2023-03-01', '2023-03-10']}
# df= pd.DataFrame(Data)
#
# # ------partition by user_id, order by order_date 取出最近三次
# # series 字符串转时间 易错df['order_date'].dt.to_datetime()
# df['order_date'] = pd.to_datetime(df['order_date'])
# # 易错 order by 列写在rank()里
# df['rn']= df.groupby(df['user_id'])['order_date'].rank(ascending=False)
# df = df[df['rn']<=3]
#
# # ------在分区内，错开加减 lag()
# # 易错 groupby没聚合没apply直接排序
# df_sorted = df.groupby('user_id')\
#               .apply(lambda x: x.sort_values('rn', ascending=False))\
#               .reset_index(drop=True)
# # 计算日期差，可以在排序后添加
# # diff()当前-前一个 ，但sql datediff(前-后 lag)
# df_sorted['days_diff'] = df_sorted.groupby('user_id')['order_date']\
#                         .diff().dt.days #  计算当前行与前一行差值(日期是升序）
# print(df_sorted)

# for user_id, group in df.groupby('user_id'):
#     group_sorted = group.sort_values('rn',ascending=True)
#     print(user_id,group_sorted)
# data =pd.read_excel('三阶段.xlsx')
'''淘宝三阶段 字典列表用DF的转换'''
# items = [
#     {
#         'session_id': 'sess1001',
#         'user_id': 'user305',
#         'product_id': 'prod_milk',
#         'add_time': '2023/6/1 9:05:23',
#         'is_daily_necessity': True,
#         'price': 5.99,
#         'discount': 0.0,
#         'in_recommend': True,
#         'category': 'dairy',          # 新增推断字段
#         'original_price': 5.99,      # 新增计算字段
#         'is_flash_sale': False       # 新增默认字段
#     },
#     {
#         'session_id': 'sess1001',
#         'user_id': 'user305',
#         'product_id': 'prod_bread',
#         'add_time': '2023/6/1 9:15:23',
#         'is_daily_necessity': False,
#         'price': 3.49,
#         'discount': 0.1,
#         'in_recommend': False,
#         'category': 'bakery',
#         'original_price': 3.88,      # 3.49/(1-0.1)≈3.88
#         'is_flash_sale': False
#     },
#     {
#         'session_id': 'sess1001',
#         'user_id': 'user305',
#         'product_id': 'prod_phonecase',
#         'add_time': '2023/6/1 9:05:20',
#         'is_daily_necessity': False,
#         'price': 15.99,
#         'discount': 0.3,
#         'in_recommend': False,
#         'category': 'electronics',
#         'original_price': 22.84,     # 15.99/(1-0.3)≈22.84
#         'is_flash_sale': True        # 折扣>0.2视为闪购
#     },
#     {
#         'session_id': 'sess1002',
#         'user_id': 'user305',
#         'product_id': 'prod_eggs',
#         'add_time': '2023/6/2 9:05:23',
#         'is_daily_necessity': True,
#         'price': 4.29,
#         'discount': 0.0,
#         'in_recommend': True,
#         'category': 'dairy',
#         'original_price': 4.29,
#         'is_flash_sale': False
#     },
#     {
#         'session_id': 'sess1002',
#         'user_id': 'user305',
#         'product_id': 'prod_toothpaste',
#         'add_time': '2023/6/1 9:15:23',
#         'is_daily_necessity': True,
#         'price': 3.79,
#         'discount': 0.0,
#         'in_recommend': True,
#         'category': 'hygiene',
#         'original_price': 3.79,
#         'is_flash_sale': False
#     },
#     {
#         'session_id': 'sess1002',
#         'user_id': 'user305',
#         'product_id': 'prod_headphones',
#         'add_time': '2023/6/1 9:25:23',
#         'is_daily_necessity': False,
#         'price': 89.99,
#         'discount': 0.15,
#         'in_recommend': False,
#         'category': 'electronics',
#         'original_price': 105.87,    # 89.99/(1-0.15)≈105.87
#         'is_flash_sale': False       # 高单价商品不标记为闪购
#     }
# ]
#
# # row_number()over() 排序，添加字position
# data=pd.DataFrame(items) # 字典列表转成数据框
# data['add_time'] = pd.to_datetime(data['add_time'],format='%Y/%m/%d %H:%M:%S')  # 会保留完整的时间信息
# data['position'] = data.groupby('session_id')['add_time'].rank(method='first').astype(int)
#
# # 先验知识定义基础规则（必需品默认标记）+ 动态调整规则
# def taobao_stage(item):
#     if item['is_daily_necessity']: # 日用品
#         return 'anchor'
#     elif item['position']<=2 and item['price']<100:  # 前2件的低价品
#         return 'anchor'
#     elif item['in_recommend'] and item['discount']>0.5: # 高折扣推荐品
#         return 'impulse'
#     else: # 将商品位置>5 ->归类为冲动品（浏览时间越长，添加的位置越往后，不是必需品和补充品（计划内）的可能越高-临时起意）
#         return 'supplement' if item['position']<=5 else 'impulse' # 三元条件
# # 将原本设计用于处理字典的函数taobao_stage,可以直接处理DataFrame的行。
# # 因为两者都支持[]访问方式。
#
# # pandas里面的apply可以沿着DataFrame的轴应用函数【反着记(axis=1)操作行，结果影响行】
# # 每行作为Series传递给函数，其行为类似字典【items中的每个字典item ↔ DataFrame中的每一行row】（Series索引：列，值：行值）
# # 当axis=1时，apply()传递给函数的row对象有以下特点：row['session_id'] 字典列名访问 row.session_id属性访问
# # 函数返回结果组成新的Series
# # 结果赋值给新列stage
# data['stage'] = data.apply(lambda row:taobao_stage(row),axis=1)
# # 展示结果(不用groupby，直接两层sort_values) sort_values(by=['session_id','position'],axis=0,ascendings=[True,True])
# data_final = data.sort_values(['session_id','position']).reset_index(drop=True)
# '''
# # 如果想从DF转换成dict列表 易错：pd.to_dict(data_final)
# data_items = data_final.to_dict('records')
# # 如果想看字典列表前3行，列表角标访问左闭右开0-2 ,键item['session_id']
# for item in data_items[:3]:
#     print(f"Session: {item['session_id']}, Product: {item['product_id']}, Position: {item['position']}, Stage: {item['stage']}")
# '''
# data = data_final.groupby(['position','stage']).agg({'stage':'count'})
# print(data)


# 连续登录天数


# 读取Excel文件

# df = pd.read_excel('test.xlsx', sheet_name='Sheet2')
#
# # 提取月份信息（假设月份格式为YYYYMM）
# df['月份'] = df['月份'].astype(str)
# df['年月'] = df['月份'].str[:4] + '-' + df['月份'].str[4:]  # 转换为YYYY-MM格式
#
# # 绘制小提琴图
# plt.figure(figsize=(12, 6))
# sns.violinplot(x='年月', y='销售金额', data=df, palette='Set3')
#
# # 添加标题和标签
# plt.title('各月份用户消费金额分布（小提琴图）')
# plt.xlabel('月份')
# plt.ylabel('消费金额')
# plt.xticks(rotation=45)  # 旋转x轴标签以便更好地显示
#
# # 显示图形
# plt.tight_layout()
# plt.show()
'''pyspark'''
#
# #  对于简单的字段提取，列表推导式更简洁易懂(混）
# #  完全转换为PySpark操作不一定总是更好。数据很小且+ Python本地处理（collect() 拉取数据到本地）
# #  collect()、show()、count() 等 Action 操作才会真正触发计算（适用小数据）
# #  .collect() 返回 Python list[Row] row['大类名称']填充值，但是numpy里面.UNIQUE后是arr,需要转列表。
# categories = [row['大类名称'] for row in df.select("大类名称").distinct().collect()]
#
# categories = df["大类名称"].unique().tolist()  # 直接转列表
# categories = [x for x in df["大类名称"].unique()] # unique()是arr,["A", "B", "C"]），不是 Row 对象

# 非对称
'''preceding'''
# 对称
# import numpy as np
# import pandas as pd
#
#
# def vectorized_calculation(df):
#     # 步骤1：提取NumPy数组
#     sales = df['销售金额'].values  # 形状 (n,)
#     fixed = df['fixed'].values  # 形状 (n,)
#     # 步骤2：生成差值矩阵（利用广播）
#     # sales[:, None] 将数组变为列向量 (n,1)
#     # sales 是行向量 (1,n)
#     # 相减后得到 n×n 矩阵
#     diff = np.abs(sales[:, None] - sales)  # 形状 (n,n)
#     # 步骤3：生成阈值矩阵
#     ranges = fixed[:, None]  # 将阈值转为列向量 (n,1)
#     # 步骤4：布尔矩阵计算
#     in_range = (diff <= ranges)  # 形状 (n,n)
#     # 步骤5：排除自身（对角线置False）
#     np.fill_diagonal(in_range, False)
#     # 步骤6：按行求和
#     df['similar_users_01'] = in_range.sum(axis=1)
#     return df
#
#
# import numpy as np
# import pandas as pd
#
#
# def dynamic_range_optimized(df):
#         # 预处理：按阈值规则分组
#         # df.apply(axis=1) 按行处理df，直接将值复制给新列，优先选择df['col'] = df.apply()
#         # 值15_25，不用.loc[?,'threshold_group'] 遍历索引idx 不推荐
#         df['threshold_group'] = df.apply(
#                 lambda x: f"{x['preceding']}_{x['following']}", axis=1)
#         results = []
#         for _, group in df.groupby('threshold_group'):
#                 sales = group['sales'].values
#                 prec = group['preceding'].iloc[0]  # 组内阈值相同
#                 foll = group['following'].iloc[0]
#                 # 向量化计算(避免纯NumPy全矩阵-内存峰值数据量）
#                 # 检查每个销售值是否大于等于【其他】销售值减去 prec ...
#                 # 两个条件的与运算(&)创建了一个矩阵，表示哪些销售值落在当前销售值的 [sales-prec, sales+foll] 范围内。
#                 in_range = (
#                         (sales[:, None] >= (sales - prec)) &   # sales 网格 (n,1)
#                         (sales[:, None] <= (sales + foll)))
#                 np.fill_diagonal(in_range, False) # 布尔矩阵的对角线元素设为False
#                 group['similar_users'] = in_range.sum(axis=1) # 对每行求和，就是计算该行中有多少个 True 值
#                 results.append(group)
#         return pd.concat(results).sort_index()
#
#
# # 示例 不同种类不同阈值 + 上下限不对称
# df = pd.DataFrame({
#         'sales': [100, 95, 110, 80, 200],
#         'preceding': [15, 15, 10, 5, 30],
#         'following': [25, 25, 40, 10, 50]
# })
# res = dynamic_range_optimized(df)

# import pandas as pd
# import numpy as np
# df=pd.read_excel('iqr_mid_cat.xlsx')
# # 假设 df 已经读进来，有两列：category, IQR
# max_iqr = df['IQR'].max()
# df['scale'] = df['IQR'] / max_iqr
#
# beta_list = np.arange(0.2, 2.1, 0.1)  # 0.2, 0.3, …, 2.0
#  # 你想要的 base，可换成 0.3*median_IQR 等
#
# res = []
# for b in beta_list:
#     base = 0.3 * df['IQR']
#     df['theta'] = base * (1 + b * (0.5 - df['scale']))
#     # 过滤掉极端负值，保底 1 元,极端值跟边界值相同
#     df['theta'] = np.clip(df['theta'], 1, 25)
#
#     res.append({
#         'beta': b,
#         'min_theta': df['theta'].min(),
#         'max_theta': df['theta'].max(),
#         'mean_theta': df['theta'].mean()
#     })
#
# scan = pd.DataFrame(res)
#
# scan.to_excel('theta.xlsx')
#
'''布尔向量与loc'''
# dict = {
#     'num': [1, 2, -3, 4, 5, -11, -7, -9, -10, -1],
#     'abc': ['a', 'b', 'c', 'd', 'e', 'c', 's', 'f', 'g', 't']
# }
# df = pd.DataFrame(dict)
# # 1. 正数保留、负数倒序(索引也要反转）
# pos = df['num'] > 0
# neg = ~pos
# val_pos = df.loc[pos, 'abc']
# val_neg = df.loc[neg, 'abc'][::-1]  # 倒序
# # 用布尔向量直接做「行切片」偶数索引
# even_idx_mask=df.index % 2 ==0
# df.iloc[even_idx_mask]
#
# 负值行全部转正
# neg=df['num']<0
# df.loc[neg,'num'] = df.loc[neg,'num']*-1 # -1没括号也行
# # 等价df.loc[neg, 'num'] *= -1
#
# dict = {
#     'num': [1, 2, -3, 4, 5, -11, -7, -9, -10, -1],
#     'abc': ['a', 'b', 'c', 'd', 'e', 'c', 's', 'f', 'g', 't']
# }
# df = pd.DataFrame(dict)
# # groupby后按条件取每组的首末行(map改true/false为其他值，同时创建新行）
# df['grp'] = (df['abc']>'e').map({True:'late',False:'early'})
# first = df.groupby('grp').head(1)
# last = df.groupby('grp').tail(1)
# print(first,last)
#
#
# dict = {
#     'num': [1, 2, -3, 4, 5, -11, -7, -9, -10, -1],
#     'abc': ['a', 'b', 'c', 'd', 'e', 'c', 's', 'f', 'g', 't']
# }
# df = pd.DataFrame(dict)
# # 布尔向量 + query 混用
# is_neg = df['num'] < 0
# df.loc[~is_neg].query('num > 3') # 用 query 对剩余行做二次筛选
# # 只保留正数，其余位置填 NaN≈
# df['num'].where(df['num']>0)  # 只保留正数，其余位置填 NaN
# # 先选字母长度=1，再选数字绝对值>=5
# [df['abc'].str.len()==1][abs(df['num'])>=5] # 链式布尔索引
'''agg 聚合'''
# dict = {
#     'num': [1, 2, 1, 4, 5, 2, -7, -9, 1, -1],
#     'date':['01','02','03','04','01','04','02','08','01','05'],
#     'abc': ['a', 'b', 'a', 'd', 'b', 'c', 'd', 'a', 'b', 'c']
# }
# df = pd.DataFrame(dict)
# '''agg 聚合'''
# # 1.时间序列分析（某个商品的价格变动时间范围，价格路径）
# df.groupby('abc').agg(date_range=('date',lambda x:(x.max(),x.min())), # ()将min和max放一起
#                       price_path=('num',list))
# # 2.多列关联聚合（按照日期聚合,将num列和"对应索引的abc列"，逐元素配对打包成list）
# # 字典列表 lambda x: [{'num': n, 'abc': v} for n, v in zip(x, df.loc[x.index, 'abc'])]
# df.groupby('date').agg(paired=('num',lambda x :list(zip(x,df.loc[x.index,'abc']))))
# # 3.统计组合（一对多 num的数据分布，均值、中位数、范围）
# df.groupby('abc').agg(
#     stats=('num',
#            lambda x:{
#                'mean':x.mean(),
#                'std':x.std(),
#                'median':np.median(x),
#                'range':x.max()-x.min()
#            }))
# # 4.条件聚合（只聚合大于阈值的值）
# df.groupby('abc').agg(filter=('num',lambda x:[v for v in x if v>2]))
# # 5.分位数切割 （待看qcut边界）
# df.groupby('abc').agg(percentile=('num',lambda x:pd.cut(x,bins=2).tolist() ))# 按值范围均分
# dict = {
#     'num': [1, 2, 1, 4, 5, 2, -7, -9, 1, -1],
#     'date':['01','02','03','04','01','04','02','08','01','05'],
#     'abc': ['a', 'b', 'a', 'd', 'b', 'c', 'd', 'a', 'b', 'c']
# }
# df = pd.DataFrame(dict)
# # agg 变形（结构变换）
# # 去重
# a=df.groupby('abc').agg(unique=('num',set)) # 去重，新名不用''
# print(a)
# # 字符串拼接(将组内'有多个值'的'字符串'拼接-单个值除外了 / ','.join(x)：先x再, 例： 1   a,a,b
# b=df.groupby('num').agg(text=('abc',lambda x:','.join(x))) # 同个订单号，对多个商品
# print(b)


# from sklearn.mixture import GaussianMixture
# def auto_split(df_big, max_k=3, bic_buffer=30):
#     """
#     输入：df_big[['大类','spend']]
#     输出：labels(list[str]), splits(list[float])
#     """
#     x = df_big['销售金额'].values.reshape(-1, 1)
#     best_k, best_bic, best_model = 1, np.inf, None
#     splits = []
#
#     # 1. 扫 1~max_k
#     for k in range(1, max_k + 1):
#         gm = GaussianMixture(n_components=k, random_state=0).fit(x)
#         bic = gm.bic(x)
#         if bic < best_bic - bic_buffer:
#             best_k, best_bic, best_model = k, bic, gm
#
#     # 2. 生成标签
#     labels = [f"{df_big['category'].iloc[0]}_{band}"
#               for band in ["low", "mid", "high", "ultra"][:best_k]]
#
#     # 3. 计算切分价（谷底）
#     if best_k > 1:
#         means = np.sort(best_model.means_.flatten())
#         splits = [(means[i] + means[i + 1]) / 2
#                   for i in range(best_k - 1)]
#
#     return labels, splits
# df1=df_big_cat.loc[df_big_cat['category']=='酒饮',['category','销售金额']]
# df2=df_big_cat.loc[df_big_cat['category']=='粮油',['category','销售金额']]
# df_mid_cat.to_excel('df_mid_cat.xlsx')
#
#
# labels,splits = auto_split(df1)
# labels1,splits1 = auto_split(df2)
# print(labels,splits,labels1,splits1)
#
# # 多个group一起apply
# # df_big = df.groupby('大类名称').apply(
# #     lambda g: g['total_spend'].values
# # ).reset_index(name='spends')
# #
'''高斯混合'''
# def find_natural_breaks(cleaned_iqr, cleaned_category):
#     '''GaussianMixture (聚合一维）  会把这一串 IQR 自动看成 K 个段
#     1.gmm.means_：k个中心值（数据各自的中心（均值）是分布的最高点（山顶），断点（边界）应该在两个分布之间密度最低的地方（山谷）
#                   对于一维数据，我们通常寻找两个相邻高斯分布的交点作为分界点）。
#     2.看概率(因为软聚类，可以属于多个簇）数据点的簇概率“该点有多可能来自某个簇”：gmm.predict_proba(x)
#     3.每个 IQR 属于哪一段（0、1、2） ：labels  = gmm.predict(x) 对完整概率信息的一种简化摘要。
#     4.只保留权重显著的'聚类中心'（“该簇的全局占比”）：weights = gmm.weights_.flatten()
#        significant_means = gmm.means_[weights > 0.01].flatten() 权重是聚类中心的
#     5. weight权重只是全局占比，具体数据点归属的概率proba还取决于数据点与各簇中心的距离和分布形状
#     6. GMM 在处理异方差（不同方差）混合分布时的能力:
#        传统聚类算法对密度差异敏感,一维数据缺乏空间信息，更难定义"接近性",重复值（8.07多次出现）强化了密集区的统计显著性
#        高密集、高值、低值。[7.85, 7.90, 8.07（多个）, 8.10, 9.75（差别）]，对应的硬分配标签是 [0, 0, 1, 1, 0]，
#        拟合出的组0高斯分布的一个关键特性：它的均值低，但方差 (Variance / Spread) 可能非常大！
#        组1 有一个均值在8.07附近，方差相对较小的高斯分布(窄分布组嵌在宽分布组中间区域的结果)
# '''
#     cleaned_iqr = np.asarray(cleaned_iqr).flatten()
#     cleaned_category = np.asarray(cleaned_category)
#     '''在一维排序数组中查找自然断点（基于统计分布）'''
#     # 步骤1：分离异常值
#     q75, q25 = np.quantile(cleaned_iqr, [0.75, 0.25])
#
#     iqr = q75 - q25
#     upper_bound = q75 + 3 * iqr
#     lower_bound = q25 - 3 * iqr
#
#     normal_mask = (cleaned_iqr >= lower_bound) & (cleaned_iqr <= upper_bound)
#     anomaly_mask = (cleaned_iqr < lower_bound) | (cleaned_iqr > upper_bound)
#     # 强制将list/DataFrame转换成array,否则DF df.iloc[]才行。array[].tolist()可以转换成list
#     normal_values = cleaned_iqr[normal_mask]  # array里面根据布尔条件直接取
#     normal_names = cleaned_category[normal_mask]
#
#     anomaly_names = [n for n, m in zip(cleaned_category, anomaly_mask) if m] # 异常中类名称
#     anomaly_values = [n for n, m in zip(cleaned_iqr, anomaly_mask) if m]
#
#
#     if len(normal_values)<2:
#         raise ValueError("正常值数据量不足，无法聚类")
#     # 步骤2：主数据聚类
#     data = np.array(normal_values).reshape(-1, 1) # 将一维数组转换为二维列向量
#     # 1. 使用贝叶斯信息准则(BIC)确定最佳分组数
#     '''BIC随k增加先降后升（过拟合时惩罚项主导）
#        肘部点选择标准：成本敏感型业务，风险：可能忽略长尾
#           当 ΔBIC(k→k+1) < 阈值 时停止增加k
#           阈值 = 前一个ΔBIC的50%（经验值）
#        BIC最小: 高风险精细分类 风险：过度复杂化'''
#     bics = []
#     n_components_range = range(1, min(6, max(2,len(normal_values)))) # 至少保证range(1,2)=1，不需要去重
#
#     for n_components in n_components_range:
#         #  强制所有高斯分布有相同的"宽度" covariance_type='tied'
#         gmm = GaussianMixture(n_components=n_components, random_state=42)
#         gmm.fit(data)
#         bics.append(gmm.bic(data))
#
#     '''第一种：用肘部法则选择最佳分组数（bics个数大于3） ,找不到用"BIC最小的分组数（不管哪种，最后至少要有2组）"'''
#     # 计算下降率，找拐点 肘部法则（最少需要3个bic此能算比值）
#     # 修改后的肘部法则逻辑
#     if len(bics) >= 3:  # 至少需要3个BIC值才能计算比值 relative_deltas计算
#         deltas = np.diff(bics)
#         relative_deltas = deltas[:-1] / (deltas[1:] + 1e-10)  # 避免除零
#         elbow_point = np.where(relative_deltas > 1.5)[0] # 返回的是满足条件的索引值（0开始）， 没有 [0]，elbow_point 会是元组格式(array([1,3], dtype=int64),)
#         # 法1还是法2  elbow_point[0] 只取第一个拐点
#         optimal_components = elbow_point[0] + 2 if len(elbow_point) > 0 else np.argmin(bics) + 1
#     else: # np.argmin把数组拉成一维后，最小值在哪个位置” # 如果数据量小，不支持2个簇，要看下权重，会退化
#         optimal_components = np.argmin(bics) + 1 if len(bics) > 0 else 1  # 默认 1  # 数据量少时直接选最小
#     # 改进后的BIC选择逻辑(除肘部法则、最小bic之外，增加绝对变化量检测）
#     # if len(bics) >= 3:
#     #     # 方法1：相对变化率拐点检测
#     #     deltas = np.diff(bics)
#     #     relative_deltas = deltas[:-1] / (deltas[1:] + 1e-10)
#     #     elbow_point = np.where(relative_deltas > 1.5)[0]
#     #
#     #     # 方法2：绝对变化量检测（更稳健）
#     #     abs_deltas = np.abs(np.diff(bics))
#     #     significant_drops = np.where(abs_deltas > 0.1 * np.max(abs_deltas))[0]
#     #
#     #     # 优先使用方法1，若无拐点则使用方法2
#     #     if len(elbow_point) > 0:
#     #         optimal_components = elbow_point[0] + 2
#     #     elif len(significant_drops) > 0:
#     #         optimal_components = significant_drops[0] + 1
#     #     else:
#     #         # 选择BIC最小的分组数
#     #         optimal_components = np.argmin(bics) + 1
#     # else:
#     #     # 小数据情况处理
#     #     if len(bics) == 0:
#     #         optimal_components = 1
#     #     elif len(bics) == 1:
#     #         optimal_components = 1
#     #     else:  # len(bics) == 2
#     #         # 当只有两个BIC值时，检查下降是否显著
#     #         # if bics[1] < bics[0] and (bics[0] - bics[1]) > 5:  # 5是经验阈值
#     #         #     optimal_components = 2
#     #         # else:
#     #         #     optimal_components = 1
#     #         optimal_components = np.argmin(bics) + 1
#
#     # 强制最小分组数为1，最大不超过数据点数量
#     optimal_components = max(1, min(optimal_components, len(normal_values)))
#     # 2. 使用高斯混合模型进行聚类
#
#     gmm = GaussianMixture(n_components=optimal_components, random_state=42, max_iter=100,covariance_type='tied') # 增加迭代次数避免不收敛
#     gmm.fit(data)
#     print("GMM converged",gmm.converged_) # 是否收敛
#     labels = gmm.predict(data)  # 易错：不改变原始数据的位置，可能会出现同一个label被断开的情况。后续要根据原始数据的索引，将label顺序调好，否则影响断点切分）
#     weights = gmm.weights_.flatten()  # 聚类权重
#     valid_clusters = np.where(weights > 0.01)[0]  # 找出有效聚类（权重大于阈值） 获取索引 (array([0,1,2,3,4]),)【0】提取索引数组 array([0,1,2,3,4])
#
#     predict_proba = gmm.predict_proba(data)  # 数据点簇归属概率
#     predict_proba = np.round(predict_proba, 2)  # 处理科学计数法
#     np.set_printoptions(suppress=True, precision=2)  # 设置输出格式（禁止科学计数法）
#
#     # 3. 按数值排序确定标签，找到分组边界
#     boundaries = []
#     sorted_values = np.sort(normal_values) # 数据先排序np.sort，获取index
#     sorted_idx = np.argsort(normal_values,kind='stable') # 数组值从小到大的索引位置(相同值排序是乱序的）
#     sorted_labels = labels[sorted_idx]
#     sorted_names = normal_names[sorted_idx]
#
#     # 识别分组变化的点
#     # 标签索引有序变化时有效
#     for i in range(1, len(sorted_labels)):  # 分组变化点  len(labels) 15个点最多只有14个分界，右开
#         if sorted_labels[i] != sorted_labels[i - 1]:
#             # 取相邻值的中点作为边界
#             boundary = (sorted_values[i - 1] + sorted_values[i]) / 2  # 某分组的最后一个值与下一个分组的第一个值相加后平均
#             boundaries.append(boundary)
#
#     #  4.根据断点创建分组
#     groups = []
#     current_group = [sorted_names[0]]
#     if  len(boundaries) == 0:
#         groups.append(normal_names.tolist())
#     else:
#         for i in range(1, len(sorted_values)):
#             # 检查是否需要创建新分组
#             if any(sorted_values[i] > bp > sorted_values[i - 1] for bp in boundaries):
#                 groups.append(current_group)  # 截断后，将current_group当前存储的该类列表，整体放进groups列表，每个元组也是列表
#                 current_group = [sorted_names[i]] # 重置变量 current_group 为当前遍历至的[i]位置上的类
#             else:
#                 current_group.append(sorted_names[i])  # 每一组的列表
#         groups.append(current_group)  # 添加最后一组
#         # 再加上异常值组
#         if len(anomaly_names)>0:
#             groups.append([( name, anomaly_values[anomaly_names.index(name)]) for name in anomaly_names])
#
#
#     name_label = []
#     for n, m in zip(normal_names, labels):
#         name_label.append({
#             'names': n,
#             'labels': m
#         })
#     for j in anomaly_names:
#         name_label.append({
#             'names': j,
#             'labels': 'Outlier'
#         })
#
#     # name_label = pd.concat([name_label,pd.DataFrame(name_label)],axis=0)
#     name_label = pd.DataFrame(name_label)
#
#
#
#     return boundaries, labels, weights, valid_clusters, predict_proba, groups,name_label
#
#
# threshold = pd.read_excel("threshold.xlsx")
# # 高斯需要将空值过滤，且数据点要大于5
# category_iqrs = threshold.groupby('大类名称').agg(IQR_values=('IQR', list), category=('中类名称', list))
#
# threshold_grp = []
# group_labels = pd.DataFrame()
# for i, row in category_iqrs.iterrows():
#     # 除空 if 不能放在 for 之前, 列表推到式里不需要：(pd.isna(x),np.isnan(x))
#     iqr_array = np.array(row['IQR_values'])
#     cat_array = np.array(row['category'])
#     # 非na掩码
#     mask = ~np.isnan(iqr_array)
#     cleaned_iqr = iqr_array[mask]
#     cleaned_category = cat_array[mask]
#
#
#     if len(cleaned_iqr) < 5:
#         break_points = '不足5条数据，不聚类'
#         print(f"{i}数据不足,break_points=[]")
#     else:
#         try:
#             break_points, labels, weights, valid_clusters, predict_proba, groups ,name_label= find_natural_breaks(cleaned_iqr,cleaned_category)
#             # 分组长表
#             name_label['Virtual'] = str(i) + '-' + name_label['labels'].astype(str)
#             group_labels = pd.concat([group_labels, name_label], axis=0)
#         except Exception as e:
#             print(f"在处理{i}类时错误：{str(e)}")  # str(e)
#             break_points = ['发生错误']
#
#
#
#     threshold_grp.append(
#         {'big_cat': i,
#          'breakpoints': break_points,
#          'labels': labels,  # 分组
#          'weights': weights,  # 分组占全局权重
#          'valid_clusters': valid_clusters,  # 有效分组
#          'predict_proba': predict_proba,  # 数据点归属概率
#          'groups': groups
#          })  # 最终名称分组
#
#
# threshold_grp = pd.DataFrame(threshold_grp)
# mask = threshold_grp['breakpoints'] =='不足5条数据，不聚类'
# columns_to_keep = ['big_cat','breakpoints']
# cols_to_empty = threshold_grp.columns.difference(columns_to_keep)
# threshold_grp.loc[mask,cols_to_empty] = '' #np.nan
# threshold_grp.to_csv('threshold_grp.csv')
# group_labels.to_csv('group_labels.csv')

'''分割列'''
# import pandas as pd
#
# # 示例数据
# data = {'category': ['食品-零食', '家电-大家电']}
# df= pd.DataFrame(data)
# # 用 extract 提取并赋值给两列
# df[['大类名称', '中类名称']] = df['category'].str.extract(r'([^-\s]+)-([^-\s]+)')
# # 用 str.split() + 正则（更灵活）
# df[['主类别', '子类别']] = df['category'].str.split(r'[-]', n=1, expand=True)
# print(df)

# 单月分层（分位数/分层/统计）
''' 适合多条件、多选项的分段选择场景：np.select用法'''
# choicelist:[静态]预先计算好的值或数组列表；
# 按顺序检查条件，第一个满足的条件会生效，后续条件不再检查；

# arr = np.array([[1, 2], [3, 4]])
# # 条件列表
# condlist = [arr > 2, arr % 2 == 0]
# choicelist = [100, 200]  # 满足条件1选100，满足条件2选200
# res = np.select(condlist, choicelist, default=0)
# '''
# 动态条件 np.piecewise(x, condlist, funclist) 直接传入函数列表，处理分段逻辑；
# '''
# def expr1(x):
#     return x ** 2
# def expr2(x):
#     return np.sin(x)
# res = np.piecewise(x, [cond1, cond2], [expr1, expr2])
# '''
# 三元操作 np.where(condition,x,y) 只能处理一个条件; 可嵌套进select '''
# choicelist = [
#     np.where(x < 2, x**2, 0),    # 条件1的表达式结果
#     np.where(x % 2 == 0, np.sin(x), 0) ] # 条件2的表达式结果
'''pd.qcut 和pd.cut 标准分箱 和 值分箱'''
# data = pd.Series([10, 200, 30, 40, 15, 90, 10, 63.33, 73.3345, 73.34])
# # 分位数分（数据量的百分比）1.q列表单调递增 2.q有重复，可设置参数duplicates='drop'
# a1 = pd.qcut(data, q=[0, 0.25, 0.75, 1], labels=['低', '中', '高'])
# # 边界值 1.边界外，两端数据为nan（10和200）,如果想包含最小值include_lowest；同上1，2
# b1 = pd.cut(data, bins=[10, 50, 80, 100], labels=['低', '中', '高'],include_lowest=True)
# '''q为整数'''
# # qcut 等频分箱（每个区间数据量几乎相同） 等价列表 q=[0,0.25,0.5,0.75,1] ｜
# a2 = pd.qcut(data, q=4, labels=['低', '中', '高', '极高'])
# # cut 等宽分箱（区间宽度相同,数据量不同），(max-min)/4 均分4段(此为间隔）
# b2 = pd.cut(data, bins=4, labels=['低', '中', '高', '极高'])
# '''其他参数'''
# # 当设置 retbins=True 时，函数会返回一个元组 (tuple) 单变量接收（得到元组）res[0],res[1],也可双变量接收
# a3, q_array = pd.qcut(data, q=3, labels=['低', '中', '高'],
#                       retbins=True,  # 是否返回分箱边界
#                       precision=2,  # 控制分箱标签的显示小数位数（不影响实际分箱逻辑）
#                       duplicates='drop')
# b3, bins_array = pd.cut(data, bins=3, labels=['低', '中', '高'],
#                         right=False,  # 不包含右边界'左闭右开'（默认不写 '左开右闭' ，True 包含右边界）
#                         include_lowest=True,  # 包含最小值 （当最小值等于第一个分箱左边界时）
#                         retbins=True, precision=2, duplicates='drop')
#
# intervals = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 60), (60, 100)])  # 强制包含最小值(左开右闭）
# # 直接指定分箱的区间对象tuple([(tuple1),(tuple2)]) "值和标签一起做了" 常用【复用分箱规则】
# b4 = pd.cut(data, bins=intervals)  # "值和标签一起做了"  结果 (5, 10] nan (10, 60]...

'''求分位数成新列：apply (r/r3都可） '''
result_mid = pd.read_excel('mid_cat.xlsx', sheet_name='Sheet1')
# result_mid = result_mid.head(10)
# def thre(df):
#     # Q1,Q3 = np.percentile(df['销售金额'],[25,75])
#     return pd.Series({'Q1':np.quantile(df['销售金额'],0.25),
#                        'Q3':np.quantile(df['销售金额'],0.75)})
# r = result_mid.groupby(['virtual','销售月份']).apply(thre).reset_index()
#
# # 结果是Series，每一行[,]2个元素的字典，需要分离
# # 1.用str[0]取出第一个,易错：直接[0]取了一行出来
# r1= result_mid.groupby(['virtual','销售月份']).apply(lambda g: np.quantile(g['销售金额'],[0.25,0.75])).str[0]
# # 2.将Series每一个元素取出（列表x），取列表第一个元素（下角标x[0])
# r2= result_mid.groupby(['virtual','销售月份']).apply(lambda g: np.quantile(g['销售金额'],[0.25,0.75]))
# r22 =r2.apply(lambda x: x[0]) # 易错：若列表推导式没了索引 pd.Series([x[0] for x in r2])
#
# # 用apply 将两列全部取出 (易错：lambda要写在前面，自动会将所有g的Series拼一起的，否则报错）
# r3= result_mid.groupby(['virtual','销售月份']).apply(lambda g: pd.Series({
#                  'q1' : np.quantile(g['销售金额'],0.25),
#                  'q3' : np.quantile(g['销售金额'],0.75)})
# ).reset_index()
'''求分位数成新列：agg (） r5/6/7正确'''
# 1.= (处理列，处理函数) 可以直接 = 处理函数（连括号都不用加）
# 2.可以不用 = ，将[（新列名，处理函数）,]打包成元组，用[]括起来效果和pd.Series({'新列名':处理函数})效果一样
# r4错
# r4 = result_mid.groupby(['virtual', '销售月份']).agg(
#     q1=('销售金额', lambda g: np.quantile(g['销售金额'], 0.25)),
#     q3=('销售金额', lambda g: np.quantile(g['销售金额'], 0.75)))
# # ** 修正r4  销售金额列在lambda函数里面不用写了,也不用拎出来
# r5 = result_mid.groupby(['virtual', '销售月份']).agg(
#     q1=('销售金额', lambda x: np.quantile(x, 0.25)),
#     q3=('销售金额', lambda x: np.quantile(x, 0.75)))
#
# # r6错 Must provide 'func' or tuples of '(column, aggfunc).
# r6 = result_mid.groupby(['virtual', '销售月份']).agg(
#     q1=lambda g: np.quantile(g['销售金额'], 0.25),
#     q3=lambda g: np.quantile(g['销售金额'], 0.75))
# # 修正r6 处理列['销售金额']放在外面，lambda里面直接g
# r7 = result_mid.groupby(['virtual', '销售月份'])['销售金额'].agg(
#     q1=lambda g: np.quantile(g, 0.25),
#     q3=lambda g: np.quantile(g, 0.75)).reset_index()
#
# # 这种写法单独记忆
# r8 = result_mid.groupby(['virtual', '销售月份'])['销售金额'].agg(
#     [('Q1', lambda x: x.quantile(0.25)), ('Q3', lambda x: x.quantile(0.75))])
# r = result_mid.groupby(['virtual', '销售月份']).agg(
#     q1 = ('销售金额',lambda x : np.quantile(x,0.25)),
#     q3 = ('销售金额',lambda x : np.quantile(x,0.75)),
#     # count不是 Python 内置的全局函数（除非你导入了它）
#     # 即使导入，count() 通常用于统计某个元素在序列中出现的次数，而不是统计满足条件的元素数量
#     cnt_low =('销售金额', lambda x : sum(1 for m in x  if m < np.quantile(x,0.25))),
#     cnt_mid =('销售金额', lambda x : sum(1 for m in x  if m >= np.quantile(x,0.25) and m < np.quantile(x,0.75))),
#     cnt_high=('销售金额', lambda x : sum(1 for m in x  if m >= np.quantile(x,0.75)))
# ).reset_index()
'''什么时候用transform'''
# 每条记录与组内均值的差距
# group['sales_vs_group_avg'] = group['销售金额'] - group['销售金额'].transform('mean')
# 用组内中位数填充缺失值
# group['销售金额'] = group['销售金额'].fillna(group['销售金额'].transform('median'))

# r1= (result_mid.groupby(['virtual', '销售月份', '层级']).agg(
#                 sim=('similar_spend_users', 'mean'),
#                 cnt=('销售金额', 'count'))
#                 .assign(
#                 ratio = lambda x : x['cnt'] / x.groupby(['virtual','销售月份'])['cnt'].transform('sum')# 上翻一层求和
#                 ))
#
# gb = result_mid.groupby(['virtual', '销售月份'])
# result_mid['size'] = gb['销售金额'].transform('size')  # 只计算一次
# r2 = (
#     result_mid.groupby(['virtual', '销售月份', '层级'])
#     .agg(
#         sim=('similar_spend_users', 'mean'),
#         cnt=('销售金额', 'count'),
#         size=('size', 'first')  # 直接取第一个值（因为组内size相同）
#     )
#     .eval('ratio = cnt / size'))  # 替代 assign，执行简单数学运算（如 A + B）/ 表达式是纯列名运算且无需外部变量。
'''如果需要先做筛选再求另外列的均值等：
1.用 apply + 自定义函数（方法 1）
2.先计算条件，再分组聚合（方法 2）当需要 保留原始行数，且计算结果是 组内共享的（如分位数、均值、总和等）用transform
例如：对每个 (virtual, 销售月份) 组，计算 销售金额 的 25% 分位数（q1），并将该值 广播到【组内】所有行。
注意：因为 transform 的 x 直接是列数据（Series），不能再像 DataFrame 那样用 x['销售金额']。所以计算条件列拿出来！
但这不需要逐行使用计算结果所以不用 transform。直接纯分组即可。
什么时候用transform？ 
'''
# def calculate_metrics(group):
#     q1 = np.quantile(group['销售金额'], 0.25)
#     q3 = np.quantile(group['销售金额'], 0.75)
#
#     mask_low = group['销售金额'] < q1
#     mask_mid = (group['销售金额'] >= q1) & (group['销售金额'] < q3)
#     mask_high = group['销售金额'] >= q3
#
#     metrics = {
#         'sim_low': group.loc[mask_low, 'similar_spend_users'].mean() if mask_low.any() else np.nan,  # 处理分层里面没有人的情况
#         'sim_mid': group.loc[mask_mid, 'similar_spend_users'].mean() if mask_mid.any() else np.nan,
#         'sim_high': group.loc[mask_high, 'similar_spend_users'].mean() if mask_high.any() else np.nan,
#         'cnt_low': mask_low.sum() if mask_low.any() else np.nan,
#         'cnt_mid': mask_mid.sum() if mask_mid.any() else np.nan,
#         'cnt_high': mask_high.sum() if mask_high.any() else np.nan,
#         # mask_low.mean() 等价于 mask_low.sum() / len(group)
#         'ratio_low': f'{mask_low.mean():.2%}' if mask_low.any() else np.nan,
#         'ratio_mid': f'{mask_mid.mean():.2%}' if mask_mid.any() else np.nan,
#         # round(,2) 改成f'{:.2%}' 里面有冒号，.2f%错
#         'ratio_high': f"{mask_high.sum() / len(group) :.2%}" if mask_high.any() else np.nan}
#     return pd.Series(metrics)
# combined = result_mid.groupby(['virtual', '销售月份']).apply(calculate_metrics)
'''升级版本'''
# def calculate_metrics(group,quantiles=[0.25, 0.75], level_names=['low', 'mid', 'high']):
#     assert len(quantiles) + 1 == len(level_names)  #     参数校验 断言 /əˈsɜːrt/
#
#     boundaries = []
#     try:
#         boundaries = [np.nanquantile(group['销售金额'].dropna(), q) for q in quantiles] # for换成列表推导式
#         boundaries = np.r_[-np.inf,boundaries,np.inf]   # 改进点1：用np.r_代替拼接列表 [-np.inf] + boundaries + [+np.inf]
#     except (IndexError, ValueError):  # 空数组或全nan
#         boundaries = [-np.inf, +np.inf]
#
#     metrics = {}
#     for i, level in enumerate(level_names):
#         mask = group['销售金额'].between( # 劣势：每个循环独立mask
#             boundaries[i],
#             boundaries[i + 1],
#             inclusive='left')  # 控制边界是否[包含]端点,左闭右开 'both', 'neither'
#         group_size = max(1, len(group))  # 劣势：防零除，每循环算一次
#         metrics.update({ # 1.update() 里面还要写{}  2.处理分层里面没有人的情况
#             f'sim_{level}': group.loc[mask,'similar_spend_users'].mean() if not mask.empty else np.nan, # nan 和 0 代表的意思不一样
#             f'cnt_{level}': mask.sum() , # 没人时 nan 和 0 都行
#             # f'ratio_{level}': f'{mask.mean():.2%}' if not mask.empty else '0%'
#             f'ratio_{level}': f"{mask.sum() / group_size:.2%}"
#         })
#     return pd.Series(metrics)
#
# combined = result_mid.groupby(['virtual', '销售月份']).apply(calculate_metrics)

"""
宽表变成长表
形如 <metric>_<suffix> 通用三段式：melt → 分组/拆分 → pivot
"""
# df = pd.DataFrame({
#     'sim_low': [543.33333],
#     'sim_mid': [480.50000],
#     'sim_high': [96.33333],
#     'cnt_low': [3],
#     'cnt_mid': [4],
#     'cnt_high': [3]
# })
# def tidy_reshape(df, stub_pat=r'^(.+)_([^-]+)$'):  # 公共前缀 与 suffix 的正则 pattern
#     return ( # 括号：链式换行可读性，随意换行不需反斜杠
#         df.melt(var_name='col', value_name='val')  # 拉直后“原来列名”那一列的列命名/“值”那一列的命名
#         .assign(  # 返回一个带新列的新 DataFrame，原表不动
#             metric=lambda d: d['col'].str.extract(stub_pat, expand=True)[0], # 默认是True 返回DF 所有捕获组[0]第1列
#             level=lambda d: d['col'].str.extract(stub_pat, expand=False)[1]) # False 返回Series？?[0]是Series第1行
#         .pivot(index='level', values='val', columns='metric') # 与pivot_table 区别
#         .rename_axis(index=None, columns=None) # 给行索引起名字（None 表示去掉名字）
#     ) # 所有操作都是新DF,原dF没变
# d = tidy_reshape(df)

'''融合版本 自定义分箱 + 标准分箱都可 + 聚合agg
'''

# 自定义分箱 标准分箱作用相同，只为保留多种写法
from interval_formatters import format_interval_unified
from ast import literal_eval


class MetricCalculator:  # 当前 auto cut左闭右开 custom 左闭右开
    def __init__(self, config_path, **kwargs):  # 构造函数
        """初始化配置（配置表）完成所有数据转换，后面函数也可用==>读文件也放进来"""
        self.config_df = self._load_and_validate_config(config_path, **kwargs)

    def _load_and_validate_config(self, path, **kwargs):
        try:
            df = pd.read_excel(path, **kwargs)
        except ValueError as e:
            print(str(e))
        if not isinstance(df,
                          pd.DataFrame):  # Return whether an object is an instance of a class or of a subclass thereof
            raise ValueError("必须提供配置DataFrame")
        required_columns = {'virtual', 'mode', 'intervals', 'levels'}
        if not required_columns.issubset(df.columns):  # config.columns 集合 是否包含 required_columns 集合
            missing = required_columns - set(df.columns)  # 差集，集合-运算 比用列表推导或循环判断更简洁
            raise ValueError(f"配置表缺少必要列:{missing}")

        df = df.set_index('virtual')
        for virtual_name, row in df.iterrows(): # 每行遍历修改效率低
            mode = row['mode'].lower()  # Auto auto
            if mode in ('nan', '', 'none', 'null'):
                raise ValueError(f'{virtual_name}:分层模式不能空 auto/custom')
            if mode not in ('auto', 'custom'):
                raise ValueError(f'{virtual_name}:mode只能是分位数分箱auto 和 自定义边界值custom')
            df.loc[virtual_name, 'mode'] = mode

            '''csv/excel file的字符串格式列表转换成python可读的对象 处理levels/intervals'''
            # list()：对字符串无效，会拆分成单个字符，两个方法 1.literal_eval 2.strip().split()
            levels = row['levels']
            if pd.isna(levels) or str(levels).strip().lower() in ('nan', 'none', 'null', '', '[]'):  # 未填写单元格 + 文本型占位符
                raise ValueError(f'{virtual_name}:至少需要一个层级')  # 1.不能为空，也不能是空列表
            try:  # 字符串 -> 列表  literal_eval 会要求不能为nan，以及是可解析的字面量 否则报错
                levels = literal_eval(levels)
                df.loc[virtual_name, 'levels'] = levels
            except ValueError as e:
                print(f'{virtual_name}:levels 非合法列表：{str(e)}')

            intervals = row['intervals']
            if pd.isna(intervals) or str(intervals).strip().lower() in ('nan', 'none', 'null', '', '[]'):
                intervals = []  # 1.不能为空，但可以是空列表[]；2.增加字符串格式的[]空列表
                df.loc[virtual_name, 'intervals'] = []
            else:
                try:  # 字符串 -> 列表(且float） .strip().split() / 跳过空列表 x.strip()
                    s = intervals.strip('[]() ')  # split() 的核心功能：将一个字符串按指定分隔符拆分成多个部分，并返回由这些部分组成的【列表】。
                    intervals = [float(x.strip()) for x in s.split(',') if x.strip()]  # .strip().split()连写不行，不给遍历
                    df.loc[virtual_name, 'intervals'] = intervals
                except ValueError as e:
                    print(f'{virtual_name}:intervals{str(e)}')
                if mode == 'auto':
                    if not all(0 < q < 1 for q in intervals):  # 判断 all()
                        raise ValueError(f'{virtual_name}:auto模式下，分位数必须在(0,1)范围内，不包括0和1')

            if len(intervals) + 1 != len(levels):
                raise ValueError(f'{virtual_name}:需要levels数量比intervals多1个 len(levels) == len(intervals) + 1')
            if sorted(intervals) != intervals:  # 检查数据是否有序（非严格升序，即允许相邻元素相等）空列表不报错
                raise ValueError(f'{virtual_name}:intervals必须已排序')

        return df

    # 函数内self拿掉，不再依赖实例变量，而是完全通过参数接收所需数据
    def _cut_binning(self, group, quantiles, levels):  # 标准分箱，"qcut只左开右闭，换cut左闭右开"
        try:
            bins = group['销售金额'].quantile(np.r_[0, quantiles, 1]).unique()
            bins[-1] = bins[-1] + 1e-10  # 添加微小增量确保包含最大值，否则"右开"最大值算超出边界
            # 不可以分箱情况：至少2个边界才能生成1个区间
            if len(bins) == 1:
                print(
                    f"警告：分组{group.name}数据无法分箱（长度={len(group)}，唯一值={group['销售金额'].nunique()}）")  # 标量
                return group.assign(intervals=f"{bins[0]:.2f}", levels='未知'), lambda: {}  # .2科学计数法 .2f小数点计数｜返回空函数
            # 可以分箱包括2类（正常 + 重复边界导致label数量与初始的不符）
            else:
                cut_result = pd.cut(group['销售金额'], bins=bins, right=False, duplicates='drop',
                                    include_lowest=True)  # 这列格式 categorical
                group = group.assign(intervals=cut_result.apply(format_interval_unified))  # 取原区间：astype(str)保留原精度括号等

                # 内部函数：处理标签的映射关系（后续聚合后处理映射）调用需返回
                def mapping_func():
                    sorted_intervals = sorted(cut_result.cat.categories,
                                              key=lambda x: x.left)  # interval格式，将分箱区间按左边界（x.left）升序排
                    # 使用外部格式化工具格式化每个区间
                    formatted_intervals = [format_interval_unified(interval) for interval in sorted_intervals]
                    if len(bins) == len(quantiles) + 2:  # (正常分组）1首1尾
                        return {interval_str: level for interval_str, level in zip(formatted_intervals, levels)}
                    else:  # （降级分组） 如果边界数不足以支持原始分箱（例如因为数据重复导致合并）
                        return {interval_str: f"特殊分组_{i + 1}" for i, interval_str in enumerate(formatted_intervals)}
            return group, mapping_func
        except Exception as e:
            print(
                f"分箱失败：分组={group.name}，错误={str(e)}")  # 刚进来groupby的group有name(分组键）,当增加新列后就没有
            return group.assign(intervals='未知', levels='未知'), lambda: {}

    def _custom_binning(self, group, boundaries, levels):  # 自定义 select可处理非连续区间，cut处理的是连续
        try:
            boundaries = np.r_[-np.inf, boundaries, np.inf]
        except:
            boundaries = [-np.inf, np.inf]

        conds = []
        mapping = {}
        for i, level in enumerate(levels):
            conds.append(group['销售金额'].between(
                boundaries[i],
                boundaries[i + 1],
                inclusive='left'))  # 左闭右开
            # 遍历添加结果到字典的方法 1.直接添加用[],类pd,mapping[label] = f"[{boundaries[i]},{boundaries[i+1]})"
            mapping.update({
                level: f"[{boundaries[i]},{boundaries[i + 1]})"
            })

        return group.assign(levels=np.select(conds, levels, default=np.nan)), mapping

    def calculate(self, group):
        ''' 自动选择对应virtual的配置'''
        if group.empty:
            return pd.DataFrame(columns=['virtual', '销售月份', 'intervals', 'levels', 'avg_sim', 'cnt', 'ratio'])

        virtual_name = group.name[0] if isinstance(group.name,tuple) else group.name # 多级索引tuple，1级索引不用[0]
        try:  #  直接读取已转换的数据
            virtual_config = self.config_df.loc[virtual_name]
        except KeyError:
            raise ValueError(f'配置表中找不到{virtual_name}的配置')

        if virtual_config['mode'] == 'auto':
            binned, mapping_func = self._cut_binning(group, quantiles=virtual_config['intervals'],
                                                     levels=virtual_config['levels'])  # 必须通过self引用实例方法 mapping为内部函数
            gb = binned.groupby(['virtual', '销售月份', 'intervals'])
            # 处理第二个返回 映射函数。为每个分组单独存储映射关系，并在最后reset_index()后根据分组来映射。
            mapping_dict = mapping_func()  # 调用映射函数获取字典
        else:
            binned, mapping = self._custom_binning(group, boundaries=virtual_config['intervals'],
                                                   levels=virtual_config['levels'])
            gb = binned.groupby(['virtual', '销售月份', 'levels'])
            mapping_dict = mapping  # 接收列表即可，也不需要调整异常和不能分组的这些情况

        # Pandas的聚合传染机制nan ，映射放聚合后
        aggregated = (gb.agg(
            avg_sim=('similar_spend_users', lambda x: f"{x.mean():.2f}"),  # 确保sim的NaN不污染其他列
            cnt=('销售金额', 'size'))  # 用size没用len
                      .assign(  # Pandas 的 groupby 在 MultiIndex DataFrame 中会智能识别层级关系，无需手动 reset_index。
            ratio=lambda x: (x['cnt'] / x.groupby(['virtual', '销售月份'])['cnt'].transform('sum'))
            .apply(lambda v: f"{v:.2%}"))  # 1.上翻一层求和 2.对Series格式再单个元素改格式。
                      .reset_index())  # 释放索引interval为普通列
        # 映射
        if virtual_config['mode'] == 'auto':
            aggregated['levels'] = aggregated['intervals'].map(mapping_dict).astype(str).replace('nan',
                                                                                                 '未知分组')  # intervals先转换为普通字符串类型
        if virtual_config['mode'] == 'custom':
            aggregated['intervals'] = aggregated['levels'].map(mapping_dict).replace('nan', '未知分组')
        # apply效率低 group['levels'] = group.apply(lambda row : level_mapping.get(row['intervals'],'未知分组'),axis=1) # 反着 按行操作

        return aggregated.reindex(columns=['virtual', '销售月份', 'intervals', 'levels', 'cnt', 'ratio', 'avg_sim'])


# 初始化配置 和 执行逻辑
config_path = '../group_labels_adjust.xlsx'
sheet_name = 'calc_mode'
calc = MetricCalculator(config_path=config_path, sheet_name=sheet_name)
# aggregated_monthly = result_mid.groupby(['virtual', '销售月份']).apply(calc.calculate).reset_index(drop=True)
aggregated_quarterly = result_mid.groupby('virtual').apply(calc.calculate).reset_index(drop=True)

'''
cut_binning 只负责：计算分箱边界、添加原始区间列、构建映射函数
calculate 负责：调用映射函数获取字典、应用映射添加层级、执行聚合操作
'''

'''读文件时，字符串转python可识别的列表对象'''
# from ast import literal_eval
# def safe_literal_eval(x):
#     if pd.isna(x):  # lietral_eval前需要处理空值
#         return tuple([]) # 用元组满足 Pandas 的哈希要求(否则报错）
#     try:
#         res = literal_eval(str(x))
#         if isinstance(res, list):
#             return tuple(res)
#         else:
#             raise ValueError(f"解析结果不是列表，而是 {type(res)}")
#     except (ValueError, SyntaxError) as e:
#         raise ValueError(f"literal_eval转换{x}报错{str(e)}")
#
# converters={'intervals': safe_literal_eval}
# result = pd.read_excel('group_labels_adjust.xlsx', sheet_name='calc_mode',converters = converters) # 解决列表等读成字符串
# '''使用数据时按需转为列表'''
# # 通过动态属性赋值：数据已预计算、创建一个新列、 存储副本、适用需要频繁访问该列
# result.intervals_list = result['intervals'].apply(lambda x: list(x) if isinstance(x, tuple) else x)
# @property  # 装饰器方法：将方法“伪装”成属性。每次访问时实时计算、不占用存储空间、无数据副本、适用偶尔访问或大数据内存敏感
# def intervals_as_list(df):
#     return df['intervals'].apply(lambda x: list(x))
# pd.DataFrame.intervals_as_list = property(intervals_as_list) # # 装饰器方法：将其作为属性添加到DataFrame类，显式挂载
# result.intervals_as_list[0] # # 触发函数执行 获取Series的第一个元素(行）
'''unstack / pipe:set_axis / melt'''
# r1 = (
#     result_mid.groupby(['virtual', '销售月份', '层级'])
#     .agg(
#         sim=('similar_spend_users', 'mean'),
#         cnt=('销售金额', 'count'))
#     .assign(
#         size=lambda x: x.groupby(['virtual', '销售月份'])['cnt'].transform('sum'),
#         ratio=lambda x: x['cnt'] / x['size'])
#     # 原 行索引层级：(0酒饮,1日期,2层级) ｜ 列索引层级：(0指标) 元组格式
#     # 转 行索引层级：(0酒饮,1日期)｜列索引层级：(0指标,1层级) 元组格式
#     .unstack('层级')  # "层级行索引"转为"层级列索引" -- 长数据转宽数据
#     # .pipe 核心作用：传递DF或Series到函数中，无需中断链式语法 （x是上一步操作后的完整 DataFrame｜第一个参数是func）
#     .pipe(lambda x: ( # 多种类型操作可以在一个pipe里(需要额外加个括号）,需共享中间变量 pipe 封装§
#     #       # a.将多层列索引"合并"为单层: set_axis(,axis=1)    例如：多(0指标 , 1层级)  变成 单（0层级_指标）
#             x.set_axis([f'{col[1]}_{col[0]}' for col in x.columns], axis=1) , # 多变单，取出col元组的第1个元素（level0）
#     #       # a1.多层行索引合并为单层： set_index()；.index.map() 是一个对索引（Index 或 MultiIndex）进行元素级转换的方法
#             x.set_index(x.index.map(lambda idx:f'{idx[1]}_{idx[0]}')), #.map(func)：对索引中的 每个元素（这里是元组） 应用函数 func
#     #       # b.调换索引0和1的位置-2层;axis可调
#             x.swaplevel(0,1,axis=1), # 但标签名称不变
#     #       # c.重构索引-多层；axis可调
#             x.reorder_levels([1,0],axis=1),# 将第1层放到前面，第0层放到后面（多层）
#     #       # d.生成列名
#             x.set_axis(
#               [f'{col[1]}_{col[0]}' if isinstance(col, tuple) and len(col) > 1  # 情况1：多层索引。检查列名 col 是否是 元组（tuple）类型
#                else f'{x.columns.name}_{col}' if x.columns.name is not None  # 情况2：单层有name（整体名称,非列名-表头行的标题）
#               else str(col)  # 情况3：单层无name
#                for col in x.columns], axis=1))
#  ))
# r2 = (
#     result_mid.groupby(['virtual', '销售月份', '层级'])
#     .agg(sim=('similar_spend_users', 'mean'),
#          cnt=('销售金额', 'count'))
#     .unstack('层级')
#     .pipe(lambda x: x.set_axis([f'{col[1]}_{col[0]}' for col in x.columns], axis=1))
#     .reset_index() # 1.先 reset, id_vars 可用 2. 不reset,melt后索引都重置了。
#     .melt(id_vars=['virtual', '销售月份'], # 保持不动的标识列
#           var_name='层级_指标', # 多层列索引 自动拼接压缩成1列"层级_指标" | 单层列索引 直接转为行索引
#           value_name='值')
# )

# 确保 cut_result 是 Categorical
# pd.cut() 返回的是一个 Categorical 对象，包含两部分：区间定义（bins）：物理存储的实际分箱边界，标签（categories）：（如 "低", "高"）
# 可以后补打标签，用.cat.rename_categories() 修改 categories
#
