import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np

# ----------------------------------------------一. 初始化应用 --------------------------------------------
app = dash.Dash(__name__)


# -----------------------------------------------二. 生成示例数据--------------------------------------------
def generate_sample_data():
    years = [2020, 2021, 2022, 2023]
    departments = ['销售部', '市场部', '技术部', '财务部']
    categories = ['电子产品', '家居用品', '服装', '食品']

    data = []
    for year in years:
        for dept in departments:
            for cat in categories:
                data.append({
                    '年份': year,
                    '部门': dept,
                    '产品类别': cat,
                    '销售额': np.random.randint(50, 200),
                    '利润': np.random.randint(10, 50),
                    '客户数': np.random.randint(20, 100),
                    '营业成本': np.random.randint(30, 100),
                    '营业外收入': np.random.randint(5, 30)
                })
    return pd.DataFrame(data)


df = generate_sample_data()
# 示例数据2: 员工满意度调查
survey_data = pd.DataFrame({
    '部门': ['销售部', '市场部', '技术部', '财务部'] * 4,
    '季度': ['Q1'] * 4 + ['Q2'] * 4 + ['Q3'] * 4 + ['Q4'] * 4,
    '满意度': np.random.randint(60, 95, 16)
})

# -------------------------------------------- 三. 创建应用布局（筛选控件移到页签外）--------------------------------
app.layout = html.Div([
    # 3.1 页面标题
    html.H1("企业数据分析大屏", style={'textAlign': 'center'}),

    # 3.2 筛选控件行
    html.Div([
        html.Div([
            html.Label("选择年份范围:"),
            dcc.RangeSlider(
                id='year-slider',
                min=int(df['年份'].min()),  # 转换为整数
                max=int(df['年份'].max()),
                step=1,
                value=[int(df['年份'].min()), int(df['年份'].max())],
                marks={str(year): str(year) for year in sorted(df['年份'].unique())}
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("选择部门:"),
            dcc.Dropdown(
                id='dept-dropdown',
                options=[{'label': dept, 'value': dept} for dept in df['部门'].unique()],
                value=['销售部', '市场部'],
                multi=True
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ], style={'padding': '10px 5px'}),

    # 3.3 页签导航
    dcc.Tabs(id='tabs', value='tab-overview', children=[
        dcc.Tab(label='概览', value='tab-overview'),
        dcc.Tab(label='营业收入', value='tab-revenue'),
    ]),

    # 3.4 页签内容
    html.Div(id='tabs-content'),

    # 3.5 (可选）数据替换说明
    html.Div([
        html.H3("数据替换说明"),
        html.Ul([
            html.Li("销售数据: 替换代码中的 generate_sample_data() 函数或直接读取CSV/Excel"),
            html.Li("员工满意度数据: 替换代码中的 survey_data DataFrame"),
            html.Li("筛选字段: 如需更改筛选条件，修改 'year-slider' 和 'dept-dropdown' 的配置")
        ])
    ], style={'marginTop': '20px', 'padding': '10px', 'border': '1px dashed #ccc'})
])


# -------------------------------------------- 四. 页签内容回调--------------------------------------------
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    # 页签一
    if tab == 'tab-overview':
        return html.Div([
            # 第一行图表
            html.Div([
                dcc.Graph(id='sales-trend'),
                dcc.Graph(id='profit-by-dept')
            ], style={'display': 'flex', 'position': 'relative'}),
            # 中间的文字说明（居中显示，相对位置）
            html.Div([
                html.H4('必要说明', style={
                    'color': '#333',
                    'padding': '10px 20px',
                    'backgroundColor': 'rgba(255,255,255,0.9)',  # 半透明背景
                    'border': '1px solid #ccc',
                    'borderRadius': '5px',
                    'position': 'relative',
                    'top': '50%',
                    'left': '50%',
                    'transform': 'translate(-50%, -50%)',  # 居中定位
                    'textAlign': 'center'
                })
            ], style={'display': 'flex', 'marginBottom': '20px'}),  # 图表行容器

            # 第二行图表
            html.Div([
                dcc.Graph(id='category-dist'),
                dcc.Graph(id='employee-satisfaction')
            ], style={'display': 'flex'}),

            # 底层图表
            html.Div([
                html.H3("销售数据明细"),
                dash_table.DataTable(
                    id='sales-table',
                    columns=[
                        {"name": "年份", "id": "年份", "type": "numeric"},
                        {"name": "部门", "id": "部门", "type": "text", "filter_options": {"independent": True}},
                        # 关键参数，解除层级依赖
                        {"name": "产品类别", "id": "产品类别", "type": "text"},
                        {"name": "客户数", "id": "客户数", "type": "numeric"},
                        {"name": "销售额", "id": "销售额", "type": "numeric", "format": {"specifier": ".0f"}},
                        {"name": "利润", "id": "利润", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "销售率", "id": "销售率", "type": "numeric",
                         "format": {"specifier": ".2%", "locale": {"symbol": ["", "%"]}}},
                        {"name": "利润率", "id": "利润率", "type": "numeric", "format": {"specifier": ".2%"}}
                    ],
                    data=df.to_dict('records'),
                    page_size=10,  # 分页
                    sort_action="native",  # 启用表格的客户端本地排序功能
                    filter_action="native",
                    style_filter={
                        'backgroundColor': '#f5f5f5',  # 可视化筛选状态
                        'fontWeight': 'bold'
                    },
                    fixed_rows={'headers': True, 'data': 0},  # 固定表头
                    fixed_columns={'headers': True, 'data': 1},  # 固定第一列
                    style_table={  # 滚动
                        'overflowX': 'auto',  # 水平滚动
                        'maxHeight': '300px',  # 最大高度
                        'overflowY': 'auto',  # 垂直滚动
                        'width': '90%'  # 表格内容区域宽度（内部滚动）
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'minWidth': '80px', 'width': '80px', 'maxWidth': '120px',  # 宽度控制
                        'whiteSpace': 'normal',  # 文本换行
                        'textAlign': 'center',  # 文本居中
                        'padding': '5px',  # 内边距
                        'fontFamily': 'Arial, sans-serif',  # 字体
                        'fontSize': 14  # 字号
                    },
                    style_data_conditional=[  # 条件样式（如数值范围）
                        {
                            'if': {'column_id': '销售额', 'filter_query': '{销售额} > 100'},
                            'backgroundColor': '#3D9970',  # 销售额 > 100 显示绿色
                            'color': 'white'
                        }
                    ])
            ], style={'width': '100%', 'margin': '0 auto','display': 'block'}) # 改为 block 布局（覆盖默认的 flex）

        ])

    # 页签二
    elif tab == 'tab-revenue':
        return html.Div([
            # 第一行图表（2个子图）
            html.Div([
                dcc.Graph(id='revenue-trend'),
                dcc.Graph(id='cost-trend')
            ], style={'display': 'flex'}),
            # 第二行图表（2个子图）
            html.Div([
                dcc.Graph(id='revenue-by-dept'),
                dcc.Graph(id='cost-by-dept')
            ], style={'display': 'block'}),
            # 第三行图表（2个子图）
            html.Div([
                dcc.Graph(id='revenue-pie'),
                dcc.Graph(id='extra-income')
            ], style={'display': 'flex'})
        ])


# -------------------------------------------- 五. 页面内容1 回调 --------------------------------------------
@app.callback(
    [Output('sales-trend', 'figure'),
     Output('profit-by-dept', 'figure'),
     Output('category-dist', 'figure'),
     Output('employee-satisfaction', 'figure'),
     Output('sales-table', 'data')],

    [Input('year-slider', 'value'),
     Input('dept-dropdown', 'value')]
)
def update_overview_figures(selected_years, selected_depts):
    start_year = int(selected_years[0]) if selected_years[0] else df['年份'].min()
    end_year = int(selected_years[1]) if selected_years[1] else df['年份'].max()
    filtered_df = df[
        (df['年份'] >= start_year) &
        (df['年份'] <= end_year) &
        (df['部门'].isin(selected_depts))
        ]

    # 销售额趋势图
    sales_trend = px.line(
        filtered_df.groupby(['年份', '部门'])['销售额'].sum().reset_index(),
        x='年份', y='销售额', color='部门',
        title='年度销售额趋势',
        labels={'销售额': '销售额 (万元)'},
        category_orders={"年份": sorted(filtered_df['年份'].unique())}
    )
    sales_trend.update_xaxes(
        type='category',
        tickvals=sorted(filtered_df['年份'].unique()),
        ticktext=[str(year) for year in sorted(filtered_df['年份'].unique())]
    )

    # 利润对比图
    profit_by_dept = px.bar(
        filtered_df.groupby(['年份', '部门'])['利润'].sum().reset_index(),
        x='部门', y='利润', color='部门',
        title='各部门利润对比',
        labels={'利润': '利润 (万元)'}
    )

    # 产品类别分布图
    category_dist = px.pie(
        filtered_df.groupby(['年份', '部门', '产品类别'])['销售额'].sum().reset_index(),
        names='产品类别', values='销售额',
        title='产品类别销售额分布'
    )

    # 员工满意度图
    filtered_survey = survey_data[survey_data['部门'].isin(selected_depts)]
    satisfaction = px.bar(
        filtered_survey,
        x='季度', y='满意度', color='部门',
        barmode='group',
        title='员工满意度季度变化',
        labels={'满意度': '满意度 (%)'},
        range_y=[0, 100]
    )

    # 计算销售额占比（注意响应控件、无需响应筛选计算）
    # 某类别汇总，相当于分子
    filtered_df_sum1 = filtered_df.groupby(['年份', '部门', '产品类别']).agg(
        {'客户数': 'sum', '销售额': 'sum', '利润': 'sum'}).reset_index()
    #  分母总销售额
    filtered_sales_amount = filtered_df.groupby(['年份', '部门']).sum()['销售额'].reset_index()
    filtered_sales_amount = filtered_sales_amount.rename(columns={'销售额': '部门总销售额'})

    filtered_df_sum1_merged = filtered_df_sum1.merge(filtered_sales_amount, how='left', on=['年份', '部门'])
    filtered_df_sum1_merged.loc[:, '销售率'] = (
            filtered_df_sum1_merged['销售额'] / filtered_df_sum1_merged['部门总销售额'])
    filtered_df_sum1_merged.loc[:, '利润率'] = (
            filtered_df_sum1_merged['利润'] / filtered_df_sum1_merged['部门总销售额'])
    filtered_df_sum = filtered_df_sum1_merged.drop('部门总销售额', axis=1, inplace=False)

    table_data = filtered_df_sum.to_dict('records')  # 图表生成代码(表格可接受的格式-列表字典）
    return sales_trend, profit_by_dept, category_dist, satisfaction, table_data


# -------------------------------------------- 六. 页面内容2 回调
@app.callback(
    [Output('revenue-trend', 'figure'),  # 输出图id
     Output('cost-trend', 'figure'),
     Output('revenue-by-dept', 'figure'),
     Output('cost-by-dept', 'figure'),
     Output('revenue-pie', 'figure'),
     Output('extra-income', 'figure')],
    [Input('year-slider', 'value'),  # 输入筛选控件
     Input('dept-dropdown', 'value')]
)
def update_revenue_figures(selected_years, selected_depts):
    start_year = int(selected_years[0]) if selected_years[0] else df['年份'].min()
    end_year = int(selected_years[1]) if selected_years[1] else df['年份'].max()
    filtered_df = df[
        (df['年份'] >= start_year) &
        (df['年份'] <= end_year) &
        (df['部门'].isin(selected_depts))
        ]

    # 1. 营业收入趋势
    revenue_trend = px.line(
        filtered_df.groupby(['年份', '部门'])['销售额'].sum().reset_index(),
        x='年份', y='销售额', color='部门',
        title='营业收入趋势',
        labels={'销售额': '金额 (万元)'}
    )
    revenue_trend.update_xaxes(type='category')

    # 2. 营业成本趋势
    cost_trend = px.line(
        filtered_df.groupby(['年份', '部门'])['营业成本'].sum().reset_index(),
        x='年份', y='营业成本', color='部门',
        title='营业成本趋势',
        labels={'营业成本': '金额 (万元)'}
    )
    cost_trend.update_xaxes(type='category')

    # 3. 各收入对比
    revenue_by_dept = px.bar(
        filtered_df.groupby('部门')['销售额'].sum().reset_index(),
        x='部门', y='销售额', color='部门',
        title='各部门营业收入对比'
    )

    # 4. 各成本对比
    cost_by_dept = px.bar(
        filtered_df.groupby('部门')['营业成本'].sum().reset_index(),
        x='部门', y='营业成本', color='部门',
        title='各部门营业成本对比'
    )

    # 5. 收入构成饼图
    revenue_pie = px.pie(
        filtered_df.groupby('产品类别')['销售额'].sum().reset_index(),
        names='产品类别', values='销售额',
        title='营业收入构成'
    )

    # 6. 营业外收入
    extra_income = px.bar(
        filtered_df.groupby('部门')['营业外收入'].sum().reset_index(),
        x='部门', y='营业外收入', color='部门',
        title='营业外收入情况'
    )

    return revenue_trend, cost_trend, revenue_by_dept, cost_by_dept, revenue_pie, extra_income


# -------------------------------------------- 七. 运行应用--------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8054)
