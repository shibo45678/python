import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

# ----------------------------------------------一. 初始化应用 --------------------------------------------

app = dash.Dash(__name__, suppress_callback_exceptions=True)
cache = {}  # 简单的内存缓存，实际生产环境Redis


# -----------------------------------------------二. 生成全局数据(静态/低频）--------------------------------------------
# 全局数据处理（应用启动时执行一次）
def load_and_process_data():
    # 生成四个卡片数据（暂定全静态）
    card_data = [
        {
            "title": "营业收入（万元）",
            "today": 101,
            "yesterday": 12,
            "month": 13,
            "year": 123
        },
        {
            "title": "营业成本（万元）",
            "today": 101,
            "yesterday": 12,
            "month": 13,
            "year": 123
        },
        {
            "title": "利润（万元）",
            "today": 101,
            "yesterday": 12,
            "month": 13,
            "year": 123
        },
        {
            "title": "毛利率（%）",
            "today": 101,
            "yesterday": 12,
            "month": 13,
            "year": 123
        }
    ]
    # 运营数据
    operational = pd.read_excel('/Users/shibo/pythonProject1/可视化/operational_data.xlsx', sheet_name='Sheet1')
    # KPI数据
    kpi = pd.read_excel('/Users/shibo/pythonProject1/可视化/KPI_tracking.xlsx', sheet_name='Sheet1')

    return {
        'card_data': card_data,
        'operational_data': operational,
        'kpi_data': kpi,

    }


processed_data = load_and_process_data()  # 初始化数据


def process_filter_data(selected_depts):
    department = processed_data['kpi_data']['部门'].unique().tolist()
    filtered_data_kpi_target = processed_data['kpi_data'][
        processed_data['kpi_data']['部门'].isin(selected_depts)]

    filtered_data_actual = processed_data['operational_data'][
        processed_data['operational_data']['所属事业部'].isin(selected_depts)]

    current_year_month = date.today()

    # 计算目标值 （kpi_data表数据
    mask_month_target = (pd.to_datetime(filtered_data_kpi_target['时间']).dt.year == current_year_month.year) & (
            pd.to_datetime(filtered_data_kpi_target['时间']).dt.month == current_year_month.month)
    monthly_income_target = filtered_data_kpi_target['收入目标'][mask_month_target].sum()
    monthly_profit_target = filtered_data_kpi_target['利润目标'][mask_month_target].sum()

    mask_year_target = pd.to_datetime(filtered_data_kpi_target['时间']).dt.year == current_year_month.year
    yearly_income_target = filtered_data_kpi_target['收入目标'][mask_year_target].sum()
    yearly_profit_target = filtered_data_kpi_target['利润目标'][mask_year_target].sum()

    # 计算实际值 （operational_data表数据）
    mask_month_actual = (pd.to_datetime(
        filtered_data_actual['业务发生结算月份']).dt.year == current_year_month.year) & (pd.to_datetime(
        filtered_data_actual['业务发生结算月份']).dt.month == current_year_month.month)
    monthly_income_actual = filtered_data_actual['总收入'][mask_month_actual].sum() / 10000  # 本月 收入
    monthly_profit_actual = filtered_data_actual['总利润'][mask_month_actual].sum() / 10000  # 本月 利润

    mask_year__actual = pd.to_datetime(filtered_data_actual['业务发生结算月份']).dt.year == current_year_month.year
    yearly_income_actual = filtered_data_actual['总收入'][mask_year__actual].sum() / 10000  # 今年 收入
    yearly_profit_actual = filtered_data_actual['总利润'][mask_year__actual].sum() / 10000  # 今年 利润

    # 计算今日
    mask_today_actual = (pd.to_datetime(
        filtered_data_actual['业务发生结算月份']).dt.year == current_year_month.year) & (pd.to_datetime(
        filtered_data_actual['业务发生结算月份']).dt.month == current_year_month.month) & (pd.to_datetime(
        filtered_data_actual['业务发生结算月份']).dt.day == current_year_month.day)

    daily_income_actual = filtered_data_actual['总收入'][mask_today_actual].sum() / 10000
    daily_cost_actual = filtered_data_actual['总成本'][mask_today_actual].sum() / 10000
    daily_profit_actual = daily_income_actual - daily_cost_actual / 10000
    daily_profit_ratio_actual = round((daily_profit_actual / daily_income_actual), 2)

    # 计算昨日
    mask_yesterday_actual = (pd.to_datetime(
        filtered_data_actual['业务发生结算月份']).dt.year == current_year_month.year) & (pd.to_datetime(
        filtered_data_actual['业务发生结算月份']).dt.month == current_year_month.month) & (pd.to_datetime(
        filtered_data_actual['业务发生结算月份']).dt.day - 1 == current_year_month.day - 1)

    yesterday_income_actual = filtered_data_actual['总收入'][mask_yesterday_actual].sum() / 10000
    yesterday_cost_actual = filtered_data_actual['总成本'][mask_yesterday_actual].sum() / 10000
    yesterday_profit_actual = yesterday_income_actual - yesterday_cost_actual / 10000
    yesterday_profit_ratio_actual = round((yesterday_profit_actual / yesterday_income_actual), 2)

    # 按月份分组汇总成本、毛利润、利润率、收入 (显示最近12个月）
    end_date = current_year_month + MonthEnd(0)
    start_date = end_date - relativedelta(months=11)
    # 如果用start，end,freq 那么生成的日期范围不会超过end_date，虽然是6月，只能选'2025-05-31'（end_date之前的月末日期）
    date_range = pd.date_range(start_date,end_date, freq='M')
    months = [date.strftime('%Y-%m') for date in date_range]  # 将datetime 转为 字符串 X轴

    # 月份分组 带月份的DataFrame要变成对应的数值序列（即每个月的值）
    # 处理利润率分母为0 的情况(group后分组项month是索引，但reset_index()后，变成数据框，month是列）
    valid_data = filtered_data_actual[~filtered_data_actual.loc[:,'总收入'].isin([0, np.nan])]
    valid_data.loc[:, 'month'] = valid_data['业务发生结算月份'].str[0:7]

    # 不加reset_index 保持Series 操作
    monthly_grouped_rev = valid_data.groupby(['month'])['总收入'].sum().reindex(months, fill_value=0)  # 确保所有月份存在  # series sort_index ,reset 成dataframe
    # monthly_grouped_rev = monthly_grouped_rev.combine_first(pd.Series(0, index=months)).reindex(months

    monthly_grouped_cost = valid_data.groupby(['month'])['总成本'].sum().reindex(months, fill_value=0)
    monthly_grouped_profit = valid_data.groupby(['month'])['总利润'].sum().reindex(months, fill_value=0)

    monthly_grouped_profit_ratio = (monthly_grouped_profit / monthly_grouped_rev).reindex(months, fill_value=0).reset_index(name='利润率')

    monthly_grouped_rev_val = monthly_grouped_rev.reset_index()['总收入'] / 10000
    monthly_grouped_cost_val = monthly_grouped_cost.reset_index()['总成本'] / 10000
    monthly_grouped_pro_val = monthly_grouped_profit.reset_index()['总利润'] / 10000
    monthly_grouped_pro_rat_val = monthly_grouped_profit_ratio['利润率']

    return {'monthly_income_target': monthly_income_target,
            'monthly_profit_target': monthly_profit_target,
            'yearly_income_target': yearly_income_target,
            'yearly_profit_target': yearly_profit_target,
            'monthly_income_actual': monthly_income_actual,
            'monthly_profit_actual': monthly_profit_actual,
            'yearly_income_actual': yearly_income_actual,
            'yearly_profit_actual': yearly_profit_actual,
            'daily_income_actual': daily_income_actual,
            'daily_cost_actual': daily_cost_actual,
            'daily_profit_actual': daily_profit_actual,
            'daily_profit_ratio_actual': daily_profit_ratio_actual,
            'yesterday_income_actual': yesterday_income_actual,
            'yesterday_cost_actual': yesterday_cost_actual,
            'yesterday_profit_actual': yesterday_profit_actual,
            'yesterday_profit_ratio_actual': yesterday_profit_ratio_actual,
            'months': months,
            'monthly_grouped_rev': monthly_grouped_rev,
            'monthly_grouped_cost': monthly_grouped_cost,
            'monthly_grouped_profit': monthly_grouped_profit,
            'monthly_grouped_profit_ratio': monthly_grouped_profit_ratio,
            'monthly_grouped_rev_val': monthly_grouped_rev_val,
            'monthly_grouped_cost_val': monthly_grouped_cost_val,
            'monthly_grouped_pro_val': monthly_grouped_pro_val,
            'monthly_grouped_pro_rat_val': monthly_grouped_pro_rat_val

            }


# -------------------------------------------- 三. 创建应用布局（筛选控件移到页签外）--------------------------------

# 定义导航栏
nav_items = [
    {"label": "概览", "path": "/"},
    {"label": "营业收入", "path": "/revenue"},
    {"label": "营业成本", "path": "/cost"},
    {"label": "毛利率", "path": "/gross-margin"},
    {"label": "利润率", "path": "/profit-margin"}
]

# 自定义样式
styles = {
    'panel': {
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'padding': '10px',
        'marginBottom': '10px',
        'boxShadow': '0 2px 3px rgba(0,0,0,0.1)'
    },
    'title': {
        'textAlign': 'left',  # 大标题居左
        'color': '#2c3e50',
        'marginBottom': '20px',
        'fontSize': 24
    },
    'sub_title': {
        'textAlign': 'left',
        'color': '#2c3e50',
        'marginBottom': '15px',
        'fontSize': 18,
        'family': 'Arial, sans-serif'
    },
    'filter_container': {
        'backgroundColor': 'rgba(245, 245, 245, 0.5)',
        'borderRadius': '8px',
        'padding': '15px',
        'marginBottom': '20px'
    },
    'chart_bg': {
        'plot_bgcolor': 'rgba(245, 245, 245, 0.5)',  # 绘图区域和整个区域的背景颜色，它们应该放在 layout 的顶层配置中，而不是嵌套在子属性中
        'paper_bgcolor': 'rgba(255, 255, 255, 0)',  # 白底
        'margin': dict(l=58, r=40, t=40, b=48)
    },
    'color': {
        'red': '#F44336',
        'yellow': '#FDD835',
        'green': '#7CB342',
        'gray': '#F0F0F0',  # 背景灰
        'dark_gray': '#212121'
    }
}


# 创建进度条图表独立函数
def create_progress_bar(title, value, max_value, unit="万元", is_yearly=False):
    percentage = (value / max_value) * 100 if max_value != 0 else 0
    color = styles['color']['red'] if percentage < 60 \
        else styles['color']['yellow'] if percentage < 80 \
        else styles['color']['green']

    fig = go.Figure()

    # 添加背景条
    fig.add_trace(go.Bar(
        x=[percentage],
        y=[title],
        orientation='h',
        marker=dict(color=color),
        width=0.55,  # 稍微加宽
        showlegend=False,
        hoverinfo='none'
    ))

    # 添加背景线
    fig.add_shape(
        type="rect",
        x0=0, x1=100, y0=-0.4, y1=0.4,  # 调整y轴范围使底板更矮
        yref="y",
        xref="x",
        line=dict(color=styles['color']['dark_gray'], width=1),
        fillcolor=styles['color']['gray'],
        opacity=0.5
    )

    # 添加阈值线
    for threshold in [60, 80]:
        fig.add_shape(
            type="line",
            x0=threshold, x1=threshold, y0=-0.4, y1=0.4,
            line=dict(color=styles['color']['dark_gray'], width=1, dash="dot")
        )

    # 添加文本标注
    fig.add_annotation(
        x=10,  # 设置为0（最左侧）
        y=0,
        text=f"{value:.1f}{unit} ({percentage:.1f}%)",
        showarrow=False,
        xanchor='left',
        yanchor="middle",
        font=dict(size=10),  # 深灰色文字
        xshift=-10  # 向左微调位置（负值表示向左移动）
    )

    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        plot_bgcolor=styles['chart_bg']['plot_bgcolor'],  # 透明
        paper_bgcolor=styles['chart_bg']['paper_bgcolor'],
        margin=dict(l=5, r=5, t=30, b=5),  # 上边距 40 像素（为标题留空间）
        height=55,  # 图表高度为 90 像素（增加高度以便显示标题）
        title=dict(text=title, x=0.2, y=0.6, xanchor="center", font=dict(size=10))
    )  # 标题垂直位置在图表高度的 80% 处

    return fig


# 创建面积图独立函数
def create_combined_chart(months, cost_val, rev_val, pro_val, rat_val):
    # 成本面积图（底部，堆叠）
    cost_trace_fig = go.Scatter(
        x=months,
        y=cost_val,
        name='成本',
        mode='none',  # lines+markers+text
        fill='tozeroy', # 从当前 y 值填充到 y=0，形成 “从底部开始” 的面积。
        fillcolor='#FDD835',  # # 黄色 直接设置填充颜色
        # marker_color='#FDD835',
        hoverinfo='x+y+name',
        hovertext=[f'{c:.1f}' for c in cost_val],
        textposition='top center',
        textfont=dict(color='#2c3e50'),
        yaxis='y'  # 绑定左侧金额轴
    )
    # 利润面积图（顶部，堆叠）
    profit_trace_fig = go.Scatter(
        x=months,
        y=[c + p for c, p in zip(cost_val, pro_val)],
        name='利润',
        mode='none',
        fill='tonexty',
        fillcolor='#7CB342',
        # marker_color='#7CB342',  # 绿色
        hoverinfo='x+name+text', # x、名称（"利润"）和自定义文本
        hovertext=[f'{p:.1f}' for p in pro_val],# hovertext 提供自定义内容，显示实际的利润值
        textposition='top center',
        textfont=dict(color='#2c3e50'),
        yaxis='y' # 绑定左侧金额轴
    )


    # 收入标签（顶部）
    label_trace_fig = go.Scatter(
        x=months,
        y=[r + 10 for r in rev_val],  # 放在收入线上方
        mode='text',
        text=[f'收入<br>{r:.1f}' for r in rev_val],
        textposition='top center',
        textfont=dict(color='#2c3e50', size=10),
        hoverinfo='none',
        showlegend=False
    )
    # 添加利润率线图
    profit_ratio_trace_fig = go.Scatter(
        x=months,
        y=round(rat_val*100,2),
        name='利润率（%）',
        yaxis='y2',
        line=dict(color='#e74c3c', width=2),
        hoverinfo='x+y+name',
        text=[f'{pr:.1f}%' for pr in rat_val*100],
        textposition='top left',
        textfont=dict(color='#e74c3c', size=10),
        mode='lines+text')  # 显示线和文本

    # 创建面积图布局

    layout = go.Layout(
        title={'text': f'营收趋势（截至{months[0]})',
               'font': {
                   'family': styles['sub_title']['family'],
                   'size': styles['sub_title']['fontSize'],
                   'color': styles['sub_title']['color']
               },
               'x': 0,  # 对应textAlign: 'left'
               'xanchor': 'left',  # 对应textAlign: 'left'
               'y': 0.95,
               'yanchor': 'top'
               },  # 添加标题样
        yaxis=dict(title='金额（万元）',
                   tickformat='.0f',
                   range=[0, (cost_val + pro_val).max() * 1.2],  # 取成本 + 利润的最大值
                   showgrid=False),
        yaxis2=dict(
            title='利润率（%）',
            overlaying='y',
            side='right',
            range=[0, max(rat_val)*100 * 1.2],
            showgrid=False  # 关闭 x 轴网格线
        ),
        hovermode='x unified',
        xaxis=dict(
            tickangle=-45,
            type='category',
            # tickson='boundaries',  # 扩大点击区域
            # dtick=1,
            showgrid=False,  # 新增，关闭 x 轴网格线
            # range=[-0.5, len(months) - 0.5]  # 微调 x 轴范围，让两端更贴近边缘，数值可按需改
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.8,
            font=dict(size=12)
        ),
        **styles['chart_bg']
    )
    return {
        'data': [cost_trace_fig,profit_trace_fig, label_trace_fig, profit_ratio_trace_fig],
        'layout': layout
    }


# 定义各个页面的布局

def overview_layout(selected_depts):
    cards = processed_data['card_data']  # 需要修改成响应筛选
    return html.Div([
        html.Div([
            html.H1("营收指标概览", style={"margin": "0", "marginTop": "20px"}),  # 大标题
            html.Div(
                "数据来源：后台 ； 数据说明：均为不含税、单位:万元",
                style={"color": "#666", "fontSize": "15px", "marginTop": "10px"}),  # 数据来源放到标题下方

        ], style={"display": "flex",
                  "flexDirection": "column",  # 垂直方向排列，让数据来源在标题下方
                  "alignItems": "flex-start"}),  # 内容左对齐

        html.Div([
            # 进度条容器
            html.Div([
                html.Div([  # 月度完成度
                    dcc.Graph(id='monthly-income-progress'),
                    dcc.Graph(id='monthly-profit-progress'),
                    dcc.Graph(id='yearly-income-progress'),
                    dcc.Graph(id='yearly-profit-progress')
                ], style={'width': '100%', 'backgroundColor': '#FAFAFA', 'padding': '5px', 'margin': '10px',
                          'borderRadius': '10px', 'height': '220px'}),  # 非常浅的背景 # 设置固定高度
            ], style={'width': '35%', 'display': 'flex', 'marginBottom': '15px'}),
            # 卡片容器，
            html.Div([
                html.Div([
                    html.H4(card["title"], style={"whiteSpace": "nowrap", "fontSize": "15px"}),  # 标题文字不换行
                    html.P(f"今日：{card['today']}", style={"whiteSpace": "nowrap", "fontSize": "13px"}),  # 内容文字不换行
                    html.P(f"昨日：{card['yesterday']}", style={"whiteSpace": "nowrap", "fontSize": "13px"}),
                    html.P(f"本月：{card['month']}", style={"whiteSpace": "nowrap", "fontSize": "13px"}),
                    html.P(f"今年：{card['year']}", style={"whiteSpace": "nowrap", "fontSize": "13px"}),
                ], id=f"card-{i}", style={
                    "width": "150px",  # 增大卡片宽度，可根据需求调整
                    "display": "inline-block",
                    "margin": "15px 0px",
                    "border": "1px solid #e0e0e0",
                    "padding": "8px",
                    "borderRadius": "10px",
                    "backgroundColor": "#FAFAFA",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "whiteSpace": "nowrap",  # 保证卡片内整体文字不换行
                    "verticalAlign": "top"
                })
                for i, card in enumerate(cards)
            ], style={
                'width': '65%',
                "display": "flex",
                "justifyContent": "flex-end",  # 卡片居右
                "marginBottom": "20px",
                "gap": "10px"})
        ], style={"display": "flex", "gap": "20px"}),

        # 第三行 面积图 + 饼图
        html.Div([
            # 左侧图表（70%宽度）
            html.Div([
                # 合并的收入和利润图表
                html.Div([
                    dcc.Graph(
                        id='combined-chart',
                        style={'height': '480px'}  # 额外加宽，否则图片右边距离边缘太远（意外调整）
                    )
                ], style={**styles['panel'], 'height': '500px'})
            ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '10px'}),

            # 右侧饼图（30%宽度）
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='dept-pie',
                        style={'height': '480px'}
                    )
                ], style={**styles['panel'], 'height': '500px'})
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '10px'})
        ], style={'display': 'flex', 'width': '100%'})
    ])


def revenue_layout(selected_depts):
    # filtered_data = processed_data['kpi_data'][
    #     processed_data['kpi_data']['部门'].isin(selected_depts)
    #     ]
    return html.Div([html.H1("营业收入页面内容")])


def cost_layout(selected_depts):
    return html.Div([html.H1("营业成本页面内容")])


def gross_margin_layout(selected_depts):
    return html.Div([html.H1("毛利率页面内容")])


def profit_margin_layout(selected_depts):
    return html.Div([html.H1("利润率页面内容")])


# 应用布局
app.layout = html.Div([
    # 路由组件 - 监听 URL 变化
    dcc.Location(id='url', refresh=False),
    # 左侧导航栏30%
    html.Div([
        html.Ul([  # 导航栏结构无序列表 (html.Ul)
            html.Li([  # 每个菜单项是一个列表项 (html.Li)
                dcc.Link(  # 内部包含一个链接 (dcc.Link)
                    item["label"],  # 文本内容
                    href=item["path"],  # 链接路径
                    id=f'nav-link-{i}',  # 为每个导航链接创建唯一的 ID ,根据导航项在列表中的位置执行特定操作(高亮）
                    style={
                        "textDecoration": "none",  # 移除下划线
                        "color": "#333",
                        "padding": "15px 10px",  # 内边距，控制链接的点击区域大小
                        "fontSize": "14px",
                        "display": "block",  # 将链接转为块级元素，使其填充整个列表项宽度
                        "fontWeight": "bold" if item["path"] == "/" else "normal",  # 设置激活状态样式
                        "borderLeft": "3px solid #1e88e5" if item["path"] == "/" else "none",
                        "backgroundColor": "#e3f2fd" if item["path"] == "/" else "#f8f9fa"  # 设置背景色，激活项使用浅蓝色
                    }
                )
            ], style={"listStyleType": "none",  # 移除默认的列表项符号（如圆点）
                      "margin": "0", "padding": "0"})  # 消除浏览器默认边距，便于自定义样式
            for i, item in enumerate(nav_items)
            # enumerate除了获取值item外，还可以获取索引i 【 索引: 0, 值: {'label': '首页', 'path': '/'}】
        ], style={
            "width": "30%",  # 左侧导航占父容器30%
            "backgroundColor": "#f8f9fa",  # 浅灰色背景
            "height": "100vh",  # （视口高度）
            "boxShadow": "2px 0 5px rgba(0,0,0,0.1)",  # 右侧阴影，增强层次感
            "paddingTop": "20px"  # 顶部内边距，避免内容紧贴顶部
        })
    ]),
    # 内容70%
    html.Div([
        # 筛选器组件 需要置顶
        html.Div([
            html.Label("选择部门:"),
            dcc.Dropdown(
                id='dept-dropdown',
                options=[{'label': dept, 'value': dept} for dept in processed_data['kpi_data']['部门'].unique()],
                value=processed_data['kpi_data']['部门'].unique().tolist(),
                multi=True
            )
        ], style={'width': '35%', 'marginTop': '20px', 'float': 'right', 'whiteSpace': 'nowrap', 'fontSize': '13px'}),

        # 页面内容容器
        html.Div(id='page-content', style={'padding': '0px'})
    ], style={"width": "100%", "padding": "10px"})
], style={"display": "flex"})


# --------------------------------------------四、回调 - 根据 URL 切换页面# --------------------------------------------
# 分开处理多个回调 1 Dash 组件（如 html.Div） 2. 包含字典的列表（每个是导航链接的样式）

# 回调 1：处理页面内容
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('dept-dropdown', 'value')]
)
def display_page(pathname, selected_depts):
    # 根据路径和筛选值选择对应页面布局
    if pathname == '/':
        return overview_layout(selected_depts)
    elif pathname == '/revenue':
        return revenue_layout(selected_depts)
    elif pathname == '/cost':
        return cost_layout(selected_depts)
    elif pathname == '/gross-margin':
        return gross_margin_layout(selected_depts)
    elif pathname == '/profit-margin':
        return profit_margin_layout(selected_depts)
    else:
        return html.Div([html.H1(f"页面未找到: {pathname}")])


# 回调 2：处理导航样式
@app.callback(
    [Output(f'nav-link-{i}', 'style') for i in range(len(nav_items))],
    Input('url', 'pathname')
)
def update_nav_styles(pathname):
    # 更新导航栏样式，高亮当前选中的页签
    nav_styles = []
    for item in nav_items:
        base_style = {  # 常规
            "textDecoration": "none",
            "color": "#333",
            "padding": "15px 20px",
            "display": "block"
        }
        if item["path"] == pathname:
            base_style.update({  # 高亮
                "fontWeight": "bold",
                "borderLeft": "3px solid #1e88e5",  # 左侧蓝色边框，增强视觉区分度
                "backgroundColor": "#e3f2fd"
            })
        nav_styles.append(base_style)
    return nav_styles


# 回调 3：处理图表更新
@app.callback(
    [Output('monthly-income-progress', 'figure'),
     Output('monthly-profit-progress', 'figure'),
     Output('yearly-income-progress', 'figure'),
     Output('yearly-profit-progress', 'figure'),
     Output('combined-chart', 'figure')],
    [Input('dept-dropdown', 'value')]
)
def update_overview_figures(selected_depts):
    filtered_data = process_filter_data(selected_depts)

    # 创建图表-进度条
    monthly_income_fig = create_progress_bar(
        f"本月-收入-完成度",
        filtered_data['monthly_income_actual'],
        filtered_data['monthly_income_target'],
    )
    monthly_profit_fig = create_progress_bar(
        f"本月-利润-完成度",
        filtered_data['monthly_profit_actual'],
        filtered_data['monthly_profit_target']
    )
    yearly_income_fig = create_progress_bar(
        f"今年-收入-完成度",
        filtered_data['yearly_income_actual'],
        filtered_data['yearly_income_target'],
        is_yearly=True
    )
    yearly_profit_fig = create_progress_bar(
        f"今年-利润-完成度",
        filtered_data['yearly_profit_actual'],
        filtered_data['yearly_profit_target'],
        is_yearly=True
    )

    # 面积图
    months = filtered_data['months']
    cost_val = filtered_data['monthly_grouped_cost_val']
    rev_val = filtered_data['monthly_grouped_rev_val']
    pro_val = filtered_data['monthly_grouped_pro_val']
    rat_val = filtered_data['monthly_grouped_pro_rat_val']

    combined_operation_chart = create_combined_chart(months, cost_val, rev_val, pro_val, rat_val)

    # cards 数据

    return monthly_income_fig, monthly_profit_fig, yearly_income_fig, yearly_profit_fig, combined_operation_chart


if __name__ == '__main__':
    app.run(debug=True, port=8053)
