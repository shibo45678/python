import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


import plotly.express as px # 三维
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus']= False # 正确修复负号
plt.rcParams['figure.figsize']=[10,8]
plt.rcParams['figure.dpi'] = 200
import matplotlib # 清理缓存
matplotlib.use('Agg')  # 切换后端# 动态分群规则

# 模拟交易数据（用户ID、消费日期、消费金额）
np.random.seed(42)
n_users = 1000
df_orders = pd.DataFrame({
    'user_id': np.random.randint(1, 101, size=n_users),
    'order_date': pd.date_range(start = '2023-01-01',end = '2023-03-31', periods=n_users), # 最多只能指定三个
    'amount': np.random.lognormal(mean=3, sigma=0.5, size=n_users)
})

# print(df_orders)
# ------------------------------------------一、计算RFM指标（以当前日期为基准）-----------------------------------------------
current_date = df_orders['order_date'].max() + pd.Timedelta(days =1) # 统计截断的日期 数据3.31日，4.1做当前，4.20在写代码
rfm = df_orders.groupby('user_id').agg({
    'order_date':lambda x:(current_date - x.max()).days, # R 距离最后一次消费的天数
    'user_id':'count', # F
    'amount':'sum' # M
}).rename(columns = {'order_date':'R','user_id':'F','amount':'M'})
# 查看RFM分布
print(rfm.describe())
'''
# RFM分析可视化模板，包含多个子图展示R、F、M的分布特征和用户分群结果------------------------------------------
plt.figure(figsize=(18, 12))
plt.suptitle("RFM Analysis Dashboard", fontsize=16, y=1.02)

# -----------------------------------
# 子图1：单变量分布（直方图 + 箱线图）
# -----------------------------------
ax1 = plt.subplot(2, 2, 1)
sns.histplot(rfm['R'], bins=30, kde=True, color='skyblue')
plt.title("Recency (R) Distribution\n(最近消费间隔天数)")
plt.xlabel("Days Since Last Purchase")

ax2 = plt.subplot(2, 2, 2)
sns.boxplot(data=rfm[['F', 'M']], palette=['orange', 'green'])
plt.title("Frequency & Monetary Distribution\n(消费频次(F)与金额分布(M))")
plt.ylabel("Value (Log Scale)")
plt.yscale('log')  # 对数坐标轴

# -----------------------------------
# 子图2：双变量关系（散点矩阵）
# -----------------------------------
ax3 = plt.subplot(2, 2, 3)
sns.scatterplot(data=rfm, x='F', y='M', hue='R',
                palette='viridis', alpha=0.7)
plt.title("F vs M with R Gradient\n(频次-金额关系，颜色表示最近消费)")
plt.xlabel("Frequency")
plt.ylabel("Monetary (Log Scale)")
plt.yscale('log')

# -----------------------------------
# 子图3：用户分群（热力图）
# -----------------------------------
ax4 = plt.subplot(2, 2, 4)

# 计算分位数分组
rfm['R_quartile'] = pd.qcut(rfm['R'], q=4, labels=["4", "3", "2", "1"])  # 1=最近
rfm['F_quartile'] = pd.qcut(rfm['F'], q=4, labels=["1", "2", "3", "4"])  # 4=最频
rfm['M_quartile'] = pd.qcut(rfm['M'], q=4, labels=["1", "2", "3", "4"])  # 4=最高

# 创建分群矩阵
cluster_matrix = rfm.groupby(['R_quartile', 'F_quartile', 'M_quartile']).size().unstack().fillna(0)
sns.heatmap(cluster_matrix, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title("RFM Segment Matrix\n(用户分群热力图)")
plt.xlabel("M_quartile (金额分位)")
plt.ylabel("R_quartile x F_quartile")

plt.tight_layout()
plt.savefig('plot1.png', bbox_inches='tight', dpi=200)
# plt.show()

# 绘制分群占比饼图

# 动态分群规则
rfm['segment'] = np.where(
    (rfm['R_quartile'] == '1') &
    (rfm['F_quartile'] == '4') &
    (rfm['M_quartile'] == '4'),
    '冠军客户',
    np.where(
        (rfm['R_quartile'] == '1') &
        (rfm['F_quartile'].isin(['3','4'])),
        '高潜力客户',
        '一般客户'
    )
)

# 绘制分群占比饼图
plt.figure(figsize=(6,6))
rfm['segment'].value_counts().plot.pie(autopct='%1.1f%%',
                                      colors=['#ff9999','#66b3ff','#99ff99'])
plt.title("客户分群占比")
# plt.show()
plt.savefig('plot2.png', bbox_inches='tight', dpi=200)

# 绘制三维图
fig = px.scatter_3d(rfm, x='R', y='F', z='M',
                    color='segment', opacity=0.7,
                    labels={'R':'Recency', 'F':'Frequency', 'M':'Monetary'},
                    title="三维RFM分群")
fig.update_layout(margin=dict(l=0, r=0, b=0))
# fig.show()
plt.savefig('plot3.png', bbox_inches='tight', dpi=200)
'''


#-----------------------------------二、RFM分群（五分位数法）---------------------------------------------------------------
# 定义分位数区间标签
# 设置权重：R=20%, F=30%, M=50%   -- F用等频分箱，M用业务分箱
def rfm_score(x, q=5):
    return pd.qcut(x, q=q, labels=range(1, q+1), duplicates='drop')

# 计算RFM分箱得分（确保转换为整数）
rfm['R_score'] = rfm_score(rfm['R'], q=5).astype(int)  # 必须转为数值型
rfm['F_score'] = rfm_score(rfm['F'], q=5).astype(int)
rfm['M_score'] = rfm_score(rfm['M'], q=5).astype(int)



weights = {'R': 0.2, 'F': 0.3, 'M': 0.5}
rfm['RFM_weighted'] = (
    rfm['R_score'] * weights['R'] +
    rfm['F_score'] * weights['F'] +
    rfm['M_score'] * weights['M']
)

# 动态阈值 取前30%的阈值（70%分位数，因为数值越大用户价值越高）
threshold = rfm['RFM_weighted'].quantile(0.7)

# 标记高价值用户（前30%）
rfm['high_value'] = rfm['RFM_weighted'] >= threshold

# 验证高价值用户比例
print("高价值用户占比:", rfm['high_value'].mean())  # 应接近30%

# 输出结果示例
print(rfm.head())

#-----------------------------------三、合并促销数据并控制用户价-不同价值用户的促销参与分布---------------------------------------------------------------

# 促销参与数据（user_id，promo_counts,purchase)
df_promo = pd.DataFrame({
    'user_id':np.arange(1,101),
    'promo_counts':np.random.randint(1,10,size=100),
    'purchase':np.random.normal(loc=50,scale=20,size=100)
})
# 合并RFM标签
df_analysis = df_promo.merge(rfm[['high_value']],left_on='user_id',right_index=True)

# 查看不同价值用户的促销参与分布
print(df_analysis.groupby(['high_value','promo_counts'])['purchase'].mean().unstack())

#-----------------------------------四、分层回归分析-----------------------------------------------------------------------------------------------
# 方法1：分组回归（高/低价值用户分别分析）
model_high = sm.OLS.from_formula('purchase ~ promo_counts',data = df_analysis[df_analysis['high_value']]).fit()
model_low = sm.OLS.from_formula('purchase ~ promo_counts',data=df_analysis[~df_analysis['high_value']]).fit()

print("高价值用户模型")
print(model_high.summary())
print("\n低价值用户模型")
print(model_low.summary())

# 方法2：加入交互项（统一模型）
model_interation = sm.OLS.from_formula('purchase~promo_counts * high_value',data =df_analysis).fit()
print('\n 交互模型')
print(model_interation.summary())