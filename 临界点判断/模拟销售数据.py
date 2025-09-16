import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# 初始化 Faker
fake = Faker('zh_CN')

# 设置随机种子
np.random.seed(42)

# 生成 100,000 条记录
n = 100_000

# --------------------------
# 1. 用户属性数据
# --------------------------
users = pd.DataFrame({
    "user_id": [f"U{str(i).zfill(6)}" for i in range(1, n + 1)],
    "is_new_user": np.random.choice([0, 1], n, p=[0.7, 0.3]),  # 30% 新用户
    "registration_date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(n)],
    "historical_purchase_freq": np.random.poisson(lam=3, size=n),  # 历史消费频率（泊松分布）
    "avg_order_value": np.random.normal(loc=200, scale=50, size=n).round(2),  # 客单价（正态分布）
})

# 计算用户生命周期价值（LTV）：历史消费频率 * 客单价 * 随机系数
users["ltv"] = (users["historical_purchase_freq"] * users["avg_order_value"] * np.random.uniform(0.8, 1.2, n)).round(2)

# --------------------------
# 2. 促销活动数据
# --------------------------
promotions = pd.DataFrame({
    "promotion_id": [f"P{str(i).zfill(4)}" for i in range(1, 101)],  # 假设有 100 次促销
    "promotion_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(100)],
    "promotion_type": np.random.choice(["折扣", "满减"], 100, p=[0.6, 0.4]),
    "discount_rate": np.where(
        np.random.choice(["折扣", "满减"], 100, p=[0.6, 0.4]) == "折扣",
        np.random.uniform(0.05, 0.3, 100).round(2),  # 折扣率 5%~30%
        np.random.choice([50, 100, 150, 200], 100)  # 满减金额
    ),
    "target_user_group": np.random.choice(["新用户", "老用户", "高价值用户"], 100)
})
# 根据 promotion_type 设置 discount_rate
promotions["discount_rate"] = np.where( # np.where(condition, x, y)，当condition为True时返回x中的值，否则返回y中的值
    promotions["promotion_type"] == "折扣",
    np.random.uniform(0.05, 0.3, 100).round(2),  # 折扣率 5%~30%
    np.random.choice([50, 100, 150, 200], 100)   # 满减金额（整数）
)
# --------------------------
# 3. 用户行为数据（关联用户和促销）
# --------------------------
# 为每个用户随机分配参与的促销次数（0~5次）
users["promotion_participation_count"] = np.random.randint(0, 6, n)

# 生成行为数据明细
behavior_data = []
for idx, user in users.iterrows():
    if user["promotion_participation_count"] > 0:
        # 随机选择参与的促销活动
        selected_promotions = promotions.sample(user["promotion_participation_count"], replace=True)
        for _, promo in selected_promotions.iterrows():
            # 购买量/金额：受促销类型和用户属性影响
            if promo["promotion_type"] == "折扣":
                base_amount = user["avg_order_value"] * (1 + np.random.uniform(-0.1, 0.2))
                purchase_amount = base_amount * (1 - promo["discount_rate"])
            else:
                base_amount = user["avg_order_value"] * (1 + np.random.uniform(-0.1, 0.2))
                purchase_amount = max(base_amount - promo["discount_rate"], 10)  # 最低 10 元

            behavior_data.append({
                "user_id": user["user_id"],
                "promotion_id": promo["promotion_id"],
                "purchase_amount": round(purchase_amount, 2),
                "days_since_last_purchase": np.random.randint(1, 90)  # 购买间隔（1~90天）
            })

# 转换为 DataFrame
behavior_df = pd.DataFrame(behavior_data)

# --------------------------
# 4. 数据保存
# --------------------------
users.to_csv("users.csv", index=False)
promotions.to_csv("promotions.csv", index=False)
behavior_df.to_csv("behavior.csv", index=False)


# 如果需要调整参数（如客单价分布、促销频率等），可修改代码中的随机分布参数。