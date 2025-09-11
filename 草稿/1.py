import pandas as pd

dict1 = {'wd': ['b', 'b', 'b', 'b'],
        'wv': [1.5, 1.3, 3.8, 6.2],
        'max_wd': [0.3, 0.5, 0.2, 0.6]}

dict2 = {'wd': ['a', 'a', 'a', 'a'],
        'wv': [1.5, 1.3, 3.8, 6.2],
        'max_wd': [0.3, 0.5, 0.2, 0.6]}
df1 = pd.DataFrame(dict1)
df2 = pd.DataFrame(dict2)
dict = {'col1':df1 ,'col2':df2}
dict_to_frame=pd.DataFrame(dict)

# def calculate(group):
#     sales = group['sales'].values
#     fixed = group['fixed'].values
#     '''下面的简化版本  [(布尔条件).sum()-1   ]
#         group['sim'] = [
#               ( (sales <= (s + f)) & (sales >= (s - f)) ).sum() - 1 for s, f in zip(sales, fixed) ]
#     '''
#     for idx,(s, f) in enumerate(zip(sales, fixed)): # idx是当前'组内'的索引(0,1)，但group仍然带着'原索引'0，1，2，3(全局索引）
#         count=0 # 向量 count = ((sales >= (s - f)) & (sales <= (s + f))).sum()
#         for sale in sales:
#             if (sale <= (s + f)) & (sale >= (s - f)):
#                 count +=1
#         print(f"当前组内索引 i = {idx}, 尝试写入原索引{group.index[idx]}，尝试写入group.loc[{group.index[idx]}]")
#         group.loc[group.index[idx], 'sim'] = count-1  # 正确通过组内索引，获得全局索引 group.index[i]
#     return group
# df_c = df.groupby('大类名称', sort=False).apply(calculate)

