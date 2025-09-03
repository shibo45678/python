'''OLS 拟合二项式、多项式

步骤：
导入必要的库（statsmodels, numpy, matplotlib, pandas）。
生成或加载示例数据。
二次回归：
a. 添加X²项。
b. 添加常数项（截距）。
c. 拟合模型，输出结果。
d. 绘制散点图和二次曲线。
多项式回归（以三次为例）：
a. 使用PolynomialFeatures生成三次项。
b. 添加常数项，拟合模型，输出结果。
c. 绘制三次曲线。
比较模型，讨论R平方和过拟合。'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(0, 10, 100) # np.linspace(start, stop, num)
Y = 2 * X ** 2 - 3 * X + 5 + np.random.normal(0, 20, 100)
# Y = aX² + bX + c，其中a=2，b=-3，c=5（常数项）然后加上均值为0，标准差为20的正态分布噪声。
df = pd.DataFrame({'X': X, 'Y': Y})
'''
# 二次回归
df['X_squared'] = df['X'] ** 2  # 生成2次项
X_sm_quad = sm.add_constant(df[['X', 'X_squared']])  # df[[]]
model_quad = sm.OLS(df['Y'], X_sm_quad).fit() # 第一个参数 Series或者数组
print("二次回归结果：")
print(model_quad.summary())

# 三次回归
poly = PolynomialFeatures(degree=3, include_bias=False)  # degree=3 表示三次项
X_poly = poly.fit_transform(df[['X']])  # 自动生成 X, X², X³
X_sm_poly = sm.add_constant(X_poly)
model_poly = sm.OLS(df['Y'], X_sm_poly).fit()
print("三次回归结果：")
print(model_poly.summary())
'''
'''
# 绘图  生成平滑拟合曲线：在多项式回归中，平滑曲线能直观比较不同阶数模型的拟合效果（贴合程度）
# 揭示数据的潜在趋势，同时减少噪声干扰原始数据通常包含噪声（随机波动），直接观察散点图可能难以发现整体规律。
X_fit = np.linspace(X.min(), X.max(), 100)
df_fit = pd.DataFrame({'X': X_fit})

# 二次拟合曲线
# 提取模型参数 model_quad.params 是 OLS 回归模型的系数数组，顺序为 [截距, X系数, X²系数]([0]截距项、[1]（β₁）、[2]（β₂）
df_fit['X_squared'] = df_fit['X'] ** 2
    #  计算拟合曲线的 Y 值 Y_hat
Y_pred_quad = model_quad.params[0] + model_quad.params[1] * df_fit['X'] + model_quad.params[2] * df_fit['X_squared']


# 三次拟合曲线
df_fit['X_cubed'] = df_fit['X'] ** 3
Y_pred_poly = model_poly.params[0] + model_poly.params[1] * df_fit['X'] + model_poly.params[2] * df_fit['X_squared'] + \
              model_poly.params[3] * df_fit['X_cubed']

# 绘制二次和三次拟合
plt.figure(figsize=(12, 6))
plt.scatter(df['X'], df['Y'], color='black', label='Data')

plt.plot(X_fit, Y_pred_quad, color='blue', label='Quadratic Fit') # 拟合曲线的 Y 值 Y_hat
plt.plot(X_fit, Y_pred_poly, color='red', linestyle='--', label='Cubic Fit')

plt.title("Polynomial Regression Comparison")
plt.legend()
plt.show()
'''
#
# # BP检验异方差 cov_type = 'HC3'
# from statsmodels.stats.diagnostic import het_breuschpagan
#
# bp_test = het_breuschpagan(model_quad.resid,model_quad.model.exog)
# print("Breusch-Pagan p-value:", bp_test[1])  # p<0.05则存在异方差
# bp_test = het_breuschpagan(model_poly.resid,model_poly.model.exog)
# print("Breusch-Pagan p-value:", bp_test[1])  # 三次项

# cond 多重共线性问题 中心化处理
df['X_centered'] = df['X'] - df['X'].mean()
df['X_centered_sq'] = df['X_centered'] ** 2
df['X_centered_cu'] = df['X_centered']**3
# 中心化后的二次回归
X_sm_sq_centered = sm.add_constant(df[['X_centered', 'X_centered_sq']])
model_sq_centered = sm.OLS(df['Y'], X_sm_sq_centered).fit()
print("中心化二次回归结果：")
print(model_sq_centered.summary())
# 中心化后的三次回归
poly = PolynomialFeatures(degree=3, include_bias=False)  # degree=3 表示三次项
X_poly_centered = poly.fit_transform(df[['X_centered']])  # 自动生成 X, X², X³
X_sm_poly_centered = sm.add_constant(X_poly_centered)
model_poly_centered = sm.OLS(df['Y'], X_sm_poly_centered).fit()
print("中心化三次回归结果：")
print(model_poly_centered.summary())

# 绘图  生成平滑拟合曲线：在多项式回归中，平滑曲线能直观比较不同阶数模型的拟合效果（贴合程度）
# 揭示数据的潜在趋势，同时减少噪声干扰原始数据通常包含噪声（随机波动），直接观察散点图可能难以发现整体规律。
X_fit_centered = np.linspace(df['X_centered'].min(), df['X_centered'].max(), 100)
df_fit_centered = pd.DataFrame({'X_centered': X_fit_centered})

# 二次拟合曲线
# 提取模型参数 model_quad.params 是 OLS 回归模型的系数数组，顺序为 [截距, X系数, X²系数]([0]截距项、[1]（β₁）、[2]（β₂）
df_fit_centered['X_centered_squared'] = df_fit_centered['X_centered'] ** 2
    #  计算拟合曲线的 Y 值 Y_hat
Y_pred_quad = model_sq_centered.params[0] + model_sq_centered.params[1] * df_fit_centered['X_centered'] + model_sq_centered.params[2] * df_fit_centered['X_centered_squared']


# 三次拟合曲线
df_fit_centered['X_cubed'] = df_fit_centered['X_centered'] ** 3
Y_pred_poly = model_poly_centered.params[0] + model_poly_centered.params[1] * df_fit_centered['X_centered'] + model_poly_centered.params[2] * df_fit_centered['X_centered_squared'] + \
              model_poly_centered.params[3] * df_fit_centered['X_cubed']

# 绘制二次和三次拟合
plt.figure(figsize=(12, 6))
plt.scatter(df['X_centered'], df['Y'], color='black', label='Data')

plt.plot(X_fit_centered, Y_pred_quad, color='blue', label='Quadratic Fit') # 拟合曲线的 Y 值 Y_hat
plt.plot(X_fit_centered, Y_pred_poly, color='red', linestyle='--', label='Cubic Fit')

plt.title("Polynomial Regression Comparison")
plt.legend()
plt.show()
