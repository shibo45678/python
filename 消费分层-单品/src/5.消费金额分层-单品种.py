import pandas as pd
import numpy as np
from scipy.stats import mstats
from sklearn.mixture import GaussianMixture
from ast import literal_eval
from interval_formatters import format_interval_unified  # 标准分箱转换类型
from pathlib import Path
from typing import TypedDict, Union  # 类型验证

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent # /Users/shibo/pythonProject1/消费分层-单品
# 构建文件路径
input_excel1 = PROJECT_ROOT / "data" / "input" / "RFM.xlsx"
input_excel2 = PROJECT_ROOT / "data" / "input" / "group_labels_adjust.xlsx"
output_excel = PROJECT_ROOT / "data" / "output" / "group_labels.xlsx"

orders = pd.read_excel(input_excel1, sheet_name='input',
                       usecols=['顾客编号', '大类名称', '中类名称', '商品类型', '规格型号', '销售金额', '销售日期',
                                '商品编码', '销售数量', '商品单价', '是否促销'],
                       dtype={'销售日期': 'string', '销售金额': 'float'})
print(orders.head(5))
print(orders.describe())  # 只数值型 不包含（字符串、分类数据）

# 提取月份信息（转换为YYYY-MM格式）
orders['销售日期'] = orders['销售日期'].astype(str)
orders['销售月份'] = orders['销售日期'].str[:4] + '-' + orders['销售日期'].str[4:6]
orders['销售日期'] = orders['销售日期'].str[:4] + '-' + orders['销售日期'].str[4:6] + '-' + orders['销售日期'].str[6:8]
print(orders.head(5))

# 检查重复
d = orders.duplicated(keep='last')
print(d)

# 空值处理
n = orders.isna().sum()  # 统计空值
print(n)
null_row = orders[(orders['顾客编号'].isna() == True) | (orders['销售金额'].isna() == True)]  # 显示带有空值的行
Index_null = null_row.index  # 删掉 顾客编号 + 金额 都是null的行, 无法填值(其他忽略）
orders1 = orders.drop(index=Index_null)
print(orders1)
n1 = orders1.isna().sum()
print(n1)

# 除0
orders1 = orders1[orders1['销售金额'] != 0]

# 再查看下分类列
print(orders1['是否促销'].unique())  # 并不是折扣和满减 ['是' '否' '6.6ad' 9.9 'ya']
discount = orders1.loc[:, ['销售金额', '商品单价', '销售数量', '是否促销']][
    (orders1['是否促销'] != '是') & (orders1['是否促销'] != '否')]
print(discount)
# 修改折扣类型（无折扣的改为'否'，有折扣的改为'是' + 取消其他类型）
for idx, row in discount.iterrows():  # iterrows 返回2个
    if round(row['商品单价'] * row['销售数量'], 2) == row['销售金额']:
        discount.loc[idx, '是否促销'] = '否'  # 注意修改值的方式
    else:
        discount.loc[idx, '是否促销'] = '是'
        if pd.isna(row['销售数量']) and (row['销售金额'] < row['商品单价']):
            discount.loc[idx, '是否促销'] = '是'

# 合并到原数据
for idx in discount.index:
    orders1.loc[idx, '是否促销'] = discount.loc[idx, '是否促销']
    print(f"{idx}:{orders1.loc[idx, '是否促销']}")
Orders = orders1

'''
主线流程：品类 → 动态阈值 → 相似用户数 → 
高中低三层 → 统计各付费层级分布变化(高中低消费平均相似用户数变化、层级用户数占比)。
'''
df_mid_cat = Orders.groupby(['销售月份', '大类名称', '中类名称', '顾客编号'])['销售金额'].sum().reset_index()
df_mid_cat['category'] = df_mid_cat['大类名称'] + '-' + df_mid_cat['中类名称']  # 中类有重复

df_big_cat = Orders.groupby(['销售月份', '大类名称', '顾客编号']).agg({'销售金额': 'sum'}).reset_index()
df_big_cat = df_big_cat.rename(columns={'大类名称': 'category'})
Orders['category'] = '全局'
df_all = Orders

# 再除0/负值
df_big_cat = df_big_cat[df_big_cat['销售金额'] > 0]
df_mid_cat = df_mid_cat[df_mid_cat['销售金额'] > 0]


# 少于4条记录时 IQR 无意义（全局替补：子品类缺样本）
# Winsorize：上下 1 % 截尾

def calculate_IQR(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(columns=['IQR', 'cnt'])
    for category in df['category'].unique():
        subset = df[df['category'] == category]

        arr = subset['销售金额'].values
        result.loc[category, 'cnt'] = len(arr)
        if len(arr) <= 4:
            result.loc[category, 'IQR'] = np.nan
            continue
        # 上下 1 % winsorize 所有极端值已经被替换为边界值，没有需要忽略的值
        arr_winsorized = mstats.winsorize(arr, (0.01, 0.01))
        # - 实际的数据数组`.data` 属性，- 一个布尔掩码（`.mask` 属性）
        arr_clean = arr_winsorized.data  # 提取实际数据  MaskedArray 转换为普通数组
        q75, q25 = np.percentile(arr_clean, [75, 25])
        if q75 - q25 == 0:
            result.loc[category, 'IQR'] = np.nan
        else:
            result.loc[category, 'IQR'] = q75 - q25

    return result.sort_values('IQR')


iqr_mid_cat = calculate_IQR(df_mid_cat)
# 或需要父类填补IQR null和0
iqr_big_cat = calculate_IQR(df_big_cat)

thres_big_cat = iqr_big_cat.reset_index().rename(columns={'index': 'category'})  # 索引变成普通列后，需要用rename改名
thres_mid_cat = iqr_mid_cat.reset_index().rename(columns={'index': 'category'})
# 或需要全局iqr 填补IQR null和0
iqr_all = calculate_IQR(df_all)
thres_all = iqr_all.iloc[0, 0]
# 分割大类-中类列
thres_mid_cat[['大类名称', '中类名称']] = thres_mid_cat['category'].str.extract(r'([^-\s]+)-([^-\s]+)')
mapping = orders[['大类名称', '中类名称']].drop_duplicates()
threshold = thres_mid_cat.merge(thres_big_cat.rename(columns={'category': '大类名称', 'IQR': 'IQR_大类'}),
                                on='大类名称', how='left')
# threshold['IQR'].fillna(threshold['IQR_大类'], inplace=True) # 大类填补
# 查看NAN情况
na = threshold[threshold['IQR'].isna()]
print(na)
# 等同 threshold['IQR'].isna().mean()
print(f"数据中 NaN 比例:{threshold['IQR'].isna().sum() / len(threshold):.2%}")
threshold['IQR'].fillna(thres_all, inplace=True)

'''
数据特征：大类下各中类的IQR差异是否显著？（基本无复购周期特点）
需求：将IQR相近且业务性质相似的中类合并成虚拟组（也包括独立组）
    例： 低消费组：即时消费，单价低、高频购买
        中消费组：酒类产品，佐餐酒类，单价50-300元，中频购买
        高消费组：高价值、礼品属性或成瘾性商品，白酒单价>300元、香烟特殊品类  ，低频购买     
方案：基于IQR和业务语义的层次分组法（业务可解释、统计稳定、阈值合理降低标准差）
     1.IQR断点
     2.例外处理:
              超高频购买商品（如矿泉水）可单独分组
              季节性商品（如中秋酒礼）设置临时分组
优势：维护成本降低、稳定性提升、业务兼容（商品陈列、价格认知、营销）       
业务规则：价格带一致性（max_price/min_price < 3）购买场景一致性、品类关联性（同属饮料/酒类）
复购周期类似、渠道类似、促销感应
    '''


class GaussianResult(TypedDict):
    name_label: pd.DataFrame
    breakpoints: Union[list[float], str]  # ｜代替Union 空列表也可
    labels: np.ndarray
    weights: np.ndarray
    valid_clusters: np.ndarray
    predict_proba: np.ndarray
    groups: list
    name_label: pd.DataFrame
    big_cat:str # 函数 返回后 再加进字典也要定义


def find_natural_breaks(cleaned_iqr: np.ndarray, cleaned_category: np.ndarray) -> GaussianResult:
    cleaned_iqr = np.asarray(cleaned_iqr).flatten()
    cleaned_category = np.asarray(cleaned_category)
    '''在一维排序数组中查找自然断点（基于统计分布）'''
    # 步骤1：分离异常值
    q75, q25 = np.quantile(cleaned_iqr, [0.75, 0.25])
    iqr = q75 - q25
    upper_bound = q75 + 3 * iqr
    lower_bound = q25 - 3 * iqr

    normal_mask = (cleaned_iqr >= lower_bound) & (cleaned_iqr <= upper_bound)
    anomaly_mask = (cleaned_iqr < lower_bound) | (cleaned_iqr > upper_bound)
    # 强制将list/DataFrame转换成array,否则DF df.iloc[]才行。array[].tolist()可以转换成list
    normal_values = cleaned_iqr[normal_mask]  # array里面根据布尔条件直接取
    normal_names = cleaned_category[normal_mask]
    anomaly_values = [n for n, m in zip(cleaned_iqr, anomaly_mask) if m]
    anomaly_names = [n for n, m in zip(cleaned_category, anomaly_mask) if m]  # 异常中类名称

    if len(normal_values) < 2:
        raise ValueError("正常值数据量不足，无法聚类")

    # 步骤2：主数据聚类
    data = np.array(normal_values).reshape(-1, 1)  # 将一维数组转换为二维列向量
    # 1. 使用贝叶斯信息准则(BIC)确定最佳分组数
    '''BIC随k增加先降后升（过拟合时惩罚项主导）
       肘部点选择标准：成本敏感型业务，风险：可能忽略长尾。当 ΔBIC(k→k+1) < 阈值 时停止增加k。阈值 = 前一个ΔBIC的50%（经验值）
       BIC最小: 高风险精细分类 风险：过度复杂化'''
    bics = []
    n_components_range = range(1, min(6, max(2, len(normal_values))))  # 至少保证range(1,2)=1，分一组
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data)
        bics.append(gmm.bic(data))
    '''第一种：用肘部法则选择最佳分组数（bics个数大于3） ,找不到用"BIC最小的分组数（不管哪种，最后至少要有2组）"'''
    # 计算下降率，找拐点 肘部法则（最少需要3个bic 才能算比值）
    if len(bics) >= 3:  # relative_deltas计算
        deltas = np.diff(bics)
        relative_deltas = deltas[:-1] / (deltas[1:] + 1e-10)  # 避免除零 首尾错开 像lag
        # 返回的是满足条件的索引值（0开始）， 没有 [0]，elbow_point 会是元组格式(array([1,3], dtype=int64),)
        elbow_point = np.where(relative_deltas > 1.5)[0]  # 1.5 参考快消品
        # 法1还是法2 ， elbow_point[0] 只取第一个拐点
        optimal_components = elbow_point[0] + 2 if len(elbow_point) > 0 else np.argmin(bics) + 1
    else:  # np.argmin把数组拉成一维后，最小值在哪个位置” # 如果数据量小，不支持2个簇，要看下权重，会退化
        optimal_components = np.argmin(bics) + 1 if len(bics) > 0 else 1  # 默认 1 数据量少时直接选最小
    # 改进后的BIC选择逻辑(除肘部法则、最小bic之外，增加绝对变化量检测）
    # if len(bics) >= 3:
    #     # 方法1：相对变化率拐点检测
    #     deltas = np.diff(bics)
    #     relative_deltas = deltas[:-1] / (deltas[1:] + 1e-10)
    #     elbow_point = np.where(relative_deltas > 1.5)[0]
    #
    #     # 方法2：绝对变化量检测（更稳健）
    #     abs_deltas = np.abs(np.diff(bics))
    #     significant_drops = np.where(abs_deltas > 0.1 * np.max(abs_deltas))[0]
    #
    #     # 优先使用方法1，若无拐点则使用方法2
    #     if len(elbow_point) > 0:
    #         optimal_components = elbow_point[0] + 2
    #     elif len(significant_drops) > 0:
    #         optimal_components = significant_drops[0] + 1
    #     else:
    #         # 选择BIC最小的分组数
    #         optimal_components = np.argmin(bics) + 1
    # else:
    #     # 小数据情况处理
    #     if len(bics) == 0:
    #         optimal_components = 1
    #     elif len(bics) == 1:
    #         optimal_components = 1
    #     else:  # len(bics) == 2
    #         # 当只有两个BIC值时，检查下降是否显著
    #         # if bics[1] < bics[0] and (bics[0] - bics[1]) > 5:  # 5是经验阈值
    #         #     optimal_components = 2
    #         # else:
    #         #     optimal_components = 1
    #         optimal_components = np.argmin(bics) + 1

    # 强制最小分组数为1，最大不超过数据点数量
    optimal_components = max(1, min(optimal_components, len(normal_values)))

    # 2. 使用高斯混合模型进行聚类
    gmm = GaussianMixture(n_components=optimal_components, random_state=42, max_iter=100,  # 增加迭代次数避免不收敛
                          covariance_type='tied')  # 强制所有高斯分布有相同的"宽度" ，集中簇 不会夹在中间 协方差大的簇里面
    gmm.fit(data)
    print("GMM converged", gmm.converged_)  # 是否收敛
    labels = gmm.predict(data)  # 易错：不改变原始数据的位置，可能会出现同一个label被断开的情况。需要调整协方差。

    predict_proba = gmm.predict_proba(data)  # 数据点簇归属概率
    predict_proba = np.round(predict_proba, 2)  # 处理科学计数法
    np.set_printoptions(suppress=True, precision=2)  # 设置输出格式（禁止科学计数法）

    weights = gmm.weights_.flatten()  # 聚类权重
    valid_clusters = np.where(weights > 0.01)[0]  # 找出有效聚类（权重大于阈值）

    # 3. 按数值排序确定标签，找到分组边界
    boundaries = []
    sorted_values = np.sort(normal_values)  # 数据先排序np.sort，获取index
    sorted_idx = np.argsort(normal_values, kind='stable')  # 数组值从小到大的索引位置(相同值排序是乱序的）
    sorted_labels = labels[sorted_idx]
    sorted_names = normal_names[sorted_idx]

    # 识别分组变化的点
    # 标签索引有序变化时有效
    for i in range(1, len(sorted_labels)):  # 分组变化点  len(labels) 15个点最多只有14个分界，右开
        if sorted_labels[i] != sorted_labels[i - 1]:
            # 取相邻值的中点作为边界
            boundary = (sorted_values[i - 1] + sorted_values[i]) / 2  # 某分组的最后一个值与下一个分组的第一个值相加后平均
            boundaries.append(boundary)

    #  4.根据断点创建分组
    groups = []
    current_group = [sorted_names[0]]
    if len(boundaries) == 0:
        groups.append(normal_names.tolist())  # array变list
    else:
        for i in range(1, len(sorted_values)):
            # 检查是否需要创建新分组
            if any(sorted_values[i] > bp > sorted_values[i - 1] for bp in boundaries):
                groups.append(current_group)  # 截断后，将current_group当前存储的该类列表，整体放进groups列表，每个元组也是列表
                current_group = [sorted_names[i]]  # 重置变量 current_group 为当前遍历至的[i]位置上的类
            else:
                current_group.append(sorted_names[i])  # 每一组的列表
        groups.append(current_group)  # 添加最后一组
        # 再加上异常值组
        if len(anomaly_names) > 0:
            groups.append([(name, anomaly_values[anomaly_names.index(name)]) for name in anomaly_names])

    name_label = []
    for n, m in zip(normal_names, labels):
        name_label.append({
            'category': n,
            'labels': m
        })
    for j in anomaly_names:
        name_label.append({
            'category': j,
            'labels': j
        })

    # name_label = pd.concat([name_label,pd.DataFrame(name_label)],axis=0)
    name_label = pd.DataFrame(name_label)

    results: GaussianResult = ({'breakpoints': boundaries,
                                'labels': labels,  # 分组
                                'weights': weights,  # 分组占全局权重
                                'valid_clusters': valid_clusters,  # 有效分组
                                'predict_proba': predict_proba,  # 数据点归属概率
                                'groups': groups,
                                'name_label': name_label})
    return results


# 高斯需要将空值过滤，且数据点要大于5
category_iqrs = threshold.groupby('大类名称').agg(IQR_values=('IQR', list), category=('中类名称', list))

threshold_grp = pd.DataFrame()
group_labels = pd.DataFrame()
for i, row in category_iqrs.iterrows():
    iqr_array = np.array(row['IQR_values'])  # list->array 每个大类里面的中类
    cat_array = np.array(row['category'])
    # 除空值  布尔索引
    mask = ~np.isnan(iqr_array)
    cleaned_iqr = iqr_array[mask]
    cleaned_category = cat_array[mask]

    if len(cleaned_iqr) < 5:
        break_points = '不足5条数据，不聚类'
    else:
        try:
            results = find_natural_breaks(cleaned_iqr, cleaned_category)  # 字典
            results['big_cat'] = i  # 增加大类的键值对
            results = pd.DataFrame(results)
            # 参考表
            threshold_grp = pd.concat([threshold_grp, results], axis=0)
            #  结果表（单独处理 name_label）
            name_label = results['name_label']  # 字典键访问--> 变量是df
            name_label['virtual'] = str(i) + '-' + name_label['labels'].astype(str)
            name_label['category'] = str(i) + '-' + name_label['category']
            group_labels = pd.concat([group_labels, name_label], axis=0)
        except Exception as e:
            print(f"在处理{i}类时错误：{str(e)}")  # str(e)
            break_points = ['发生错误']

mask = threshold_grp['breakpoints'] == '不足5条数据，不聚类'
columns_to_keep = ['big_cat', 'breakpoints']
cols_to_empty = threshold_grp.columns.difference(columns_to_keep)
threshold_grp.loc[mask, cols_to_empty] = ''  # np.nan
# threshold_grp.to_csv('threshold_grp.csv') # 参考表
group_labels.to_excel(output_excel)  # 结果表

'''读取人工调整后的文件，重做IQR'''
group_labels_adjust = pd.read_excel(input_excel2, usecols=['category', 'labels', 'virtual'])
df_mid_cat_ad = (df_mid_cat.merge(group_labels_adjust, on='category', how='left')
                 .rename(columns={'category': 'big_mid_cat', 'virtual': 'category'}))  # virtual改成category进函数
df_mid_cat_ad_iqr = calculate_IQR(df_mid_cat_ad)
df_mid_cat_ad_iqr['IQR'].fillna(thres_all, inplace=True)  # 小于4为空，填充全局IQR

'''
阈值调整
把阈值上限锁在 20 元以内 完全符合快消品“客单价低、决策快、差异颗粒度小”的业务体感。
1.手动：阈值差异相对集中、快消品特性明显【低值品类扩大阈值】避免群体过度细分：1元与2元区分； 【高值品类缩小阈值】避免将不同能力混为一谈：50和60不能归为一类
对阈值<3的品类(烘焙、熟食等)考虑绝对金额范围(如±1.5元)，对战略品类可保留动态计算以捕捉细微变化
2.公式：当品类爆炸阈值的公式调整（可选）：
先对每个品类 c 计算 IQR_c，再归一化到 0-1：scale = IQR_c / max(IQR_all)
然后动态阈值 θ_c = base * (1 + b * (0.5 - scale))，其中 base 取 0.3 IQR，b 网格搜索得到最佳）。
'''


def adjust_IQR(df):
    # 简单网格搜索，选公式的b值
    # 判断标准：theta < n(n=20) 最接近20，取b的最大值（均值处于5元左右）
    max_iqr = df['IQR'].max()
    df['scale'] = df['IQR'] / max_iqr
    base = 0.3 * df['IQR']
    m, n = 1, 20
    beta_list = np.arange(0.2, 2.1, 0.1)  # 0.2, 0.3, …, 2.0

    result = []
    for b in beta_list:
        df['theta'] = base * (1 + b * (0.5 - df['scale']))
        # 过滤掉极端负值(保底 1 元,极端值跟边界值20相同)
        df['theta'] = np.clip(df['theta'], m, n)
        result.append({
            'beta': b,
            'min_theta': df['theta'].min(),
            'max_theta': df['theta'].max(),
            'mean_theta': df['theta'].mean()
        })
    scan = pd.DataFrame(result).sort_values('max_theta', ascending=True).reset_index(drop=True)

    for i in range(len(scan) - 1):  # 防止用i-1/i+1 越界
        if (scan.loc[i, 'max_theta'] < n) & (scan.loc[i + 1, 'max_theta'] == n):
            b = scan.loc[i, 'beta']
            break
        else:
            b = scan.iloc[-1]['beta']  # 最大 注意iloc[]['col']与loc[,'col']的差别
    df['adjust'] = 0.3 * (1 + b * (0.5 - df['scale']))
    df['adjust_thres'] = np.clip(base * (1 + b * (0.5 - df['scale'])), m, n)
    return df


'''向量
# 1) 找到所有 max_theta != n 的行 mask_eq = scan['max_theta'] != n
# 2) 在这些行里取最大的 beta（即最大的 b）b = scan.loc[mask_eq, 'beta'].max()
'''

adjust_thres = adjust_IQR(df_mid_cat_ad_iqr)
adjust_thres['preceding'] = adjust_thres['adjust_thres']
adjust_thres['following'] = adjust_thres['adjust_thres']
adjust_thres = adjust_thres.reset_index().rename(columns={'index': 'virtual'})

'''统计similar_user_count 先merge将阈值放进大表上成独立列，再计算（比双循环好）'''
group_labels_adjust = group_labels_adjust.rename(columns={'category': 'big_mid_cat'})
data_merged_mid = df_mid_cat_ad.rename(columns={'category': 'virtual'}) \
    .merge(adjust_thres, on='virtual', how='left')  # 用分完类的virtual连接


def range_optimized(df):
    df['threshold_group'] = df.apply(
        lambda row: f"{row['preceding']}-{row['following']}",
        axis=1
    )
    results = []
    for _, group in df.groupby(['销售月份', 'threshold_group'], sort=False):
        sales = group['销售金额'].values
        prec = group['following'].iloc[0]
        foll = group['following'].iloc[0]
        in_range: np.ndarray = ((sales[:, None] >= sales - prec) &  # 不需要显式写成 sales[None, :] - prec # 添加类型注释
                                (sales[:, None] <= (sales[None, :] + foll)))  # 因为 NumPy 会自动处理广播
        np.fill_diagonal(in_range, False)  # 不包括自己 在in_range 上面改动
        group['similar_spend_users'] = in_range.sum(axis=1)  # 对每行求和
        results.append(group)
    # results = [group1, group2, group3] pd.concat()自动保留原始索引，除非ignore_index=True
    # sort_index()按照原始索引排序
    # 结果的DF，groupby索引释放
    return pd.concat(results).sort_index(level=1)


result_mid = range_optimized(data_merged_mid)

result_mid.to_excel('mid_cat.xlsx')

'''
统计各层级分布变化
1. 单月分层(分层标签不偏移，指标是相对概念、"分位数"变化易于发现规律） 月度精准运营策略
2. 整体分层(同一"高中低"标准，各月"相似用户数均值"可比） 用户消费能力的长期变化
关注"人数"变化与"相似用户数均值"变化
'''


class MetricCalculator:  # 当前 auto cut左闭右开 custom 左闭右开
    def __init__(self, config_path, **kwargs):  # 构造函数
        """初始化配置（配置表）完成所有数据转换，后面函数也可用==> 读文件也放进来
        把公共逻辑 self.config_df 放在一个函数里，其他函数通过参数解包调用它。
        解包传递参数**kwargs:如果函数 A 需要调用函数 B，但不确定 B 需要哪些参数"""
        self.config_df = self._load_and_validate_config(config_path, **kwargs)  # 解包传递

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
        for virtual_name, row in df.iterrows():
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

        virtual_name = group.name[0] if isinstance(group.name, tuple) else group.name  # 多级索引tuple，1级索引不用[0]
        try:  # 直接读取已转换的数据
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
config_path = 'group_labels_adjust.xlsx'
sheet_name = 'calc_mode'
calc = MetricCalculator(config_path=config_path, sheet_name=sheet_name)
aggregated_monthly = result_mid.groupby(['virtual', '销售月份']).apply(calc.calculate).reset_index(drop=True)
aggregated_quarterly = result_mid.groupby('virtual').apply(calc.calculate).reset_index(drop=True)

'''
cut_binning 只负责：计算分箱边界、添加原始区间列、构建映射函数
calculate 负责：调用映射函数获取字典、应用映射添加层级、执行聚合操作
'''
