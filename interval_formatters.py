import pandas as pd
def format_interval_unified(value):
    """
    通用区间格式化工具
    支持: pandas.Interval, 数值, 字符串, None, 及其他对象
    """
    # 处理Interval对象
    if isinstance(value, pd.Interval):
        left = round(value.left, 2)
        right = round(value.right, 2)
        lb = '[' if value.closed_left else '('
        rb = ']' if value.closed_right else ')'
        return f"{lb}{left},{right}{rb}"

    # 处理数值
    if isinstance(value, (int, float)):
        return f"固定值[{value:.2f}]"

    # 处理空值
    if value is None:
        return "空值"

    # 保底返回字符串
    return str(value)