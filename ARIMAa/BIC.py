import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

# 读取数据
file_path = '深圳签订外贸合同项数数据1990~2023.xlsx'
data = pd.read_excel(file_path, sheet_name=0)
data['时间'] = pd.to_datetime(data['时间'], format='%Y')
data.set_index('时间', inplace=True)

# 选择目标列（例如：中国香港、澳门）
target_column = '其他'
series = data[target_column]

# 定义参数范围
p_range = range(0, 4)  # 自回归项阶数范围
d_range = range(0, 3)  # 差分阶数范围
q_range = range(0, 4)  # 滑动平均项阶数范围

# 初始化最佳模型和最低 BIC 值
best_bic = float("inf")
best_order = None
best_model = None

# 遍历参数组合
for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                # 拟合 ARIMA 模型
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                # 获取 BIC 值
                bic = model_fit.bic
                # 更新最佳模型
                if bic < best_bic:
                    best_bic = bic
                    best_order = (p, d, q)
                    best_model = model_fit
                # print(f"Order: {p, d, q}, BIC: {bic}")
            except Exception as e:
                continue

# 输出最佳模型的参数和 BIC 值
print(f"Best ARIMA Order: {best_order}")
print(f"Best BIC: {best_bic}")