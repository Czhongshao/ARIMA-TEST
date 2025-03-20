"""
# 深圳进出口量预测SARIMA模型

## 使用到的模块及其安装
 - Python version: 3.11.0
 - Pandas version: 2.2.3
    - pip install pandas==2.2.3
 - Matplotlib version: 3.10.1
    - pip install matplotlib==3.10.1
 - Seaborn version: 0.13.2
    - pip install seaborn==0.13.2
 - Statsmodels version:
    - pip install statsmodels==0.14.4
"""

import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', category=UserWarning)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


output_folder = "out_files"
# 创建保存excel的文件夹
excels_folder = os.path.join(output_folder, "excels")
if not os.path.exists(excels_folder):
    os.makedirs(excels_folder)
    print(f"文件夹 '{excels_folder}' 已创建。")
# 创建保存图像的文件夹
figs_folder = os.path.join(output_folder, "figs")
if not os.path.exists(figs_folder):
    os.makedirs(figs_folder)
    print(f"文件夹 '{figs_folder}' 已创建。")

## ADF 检验
# H0：它是非平稳的  
# H1：它是平稳的
def adfuller_test(series, title=""):
    """
    执行 ADF 检验并打印结果。
    
    参数:
    - series: 时间序列数据
    - title: 数据的标题，用于打印结果时更清晰地标识
    """
    print(f"\nADF 检验 - {title}")
    result = adfuller(series)
    labels = ['ADF检验统计量', 'p值', '使用的滞后数', '使用的观测值数量']
    for value, label in zip(result, labels):
        print(f"{label} : {value}")
    
    if result[1] <= 0.05:
        print("反对原假设(H0)的有力证据，否定原假设。数据没有单位根，并且是平稳的。")
    else:
        print("反对零假设的弱证据，时间序列有一个单位根，表明它是非平稳的。")

def ADFs(d, D, df_total):
    """
    对不同差分阶数的数据进行 ADF 检验。
    
    参数:
    - d: 非季节性差分阶数
    - D: 季节性差分阶数
    - s: 季节长度
    - df_total: 包含时间序列数据的 DataFrame
    """
    # 原始数据的 ADF 检验
    adfuller_test(df_total[need_to_predicted], title="原始数据")

    if d != 0:
        df_total[f'{d}阶差分'] = df_total[need_to_predicted] - df_total[need_to_predicted].shift(d)   # d阶差分
        adfuller_test(df_total[f'{d}阶差分'].dropna(), title=f'{d}阶差分')
    elif D != 0:
        df_total[f'{D}阶差分'] = df_total[need_to_predicted] - df_total[need_to_predicted].shift(D)   # 季节性D阶差分
        adfuller_test(df_total[f'{D}阶差分'].dropna(), f'{D}阶差分')

    # 经过季节性差分后的数据的 ADF 检验
    df_total['季节性差分'] = df_total[need_to_predicted] - df_total[need_to_predicted].shift(12)
    adfuller_test(df_total['季节性差分'].dropna(), title='季节性差分')

    # 绘制季节性差分图
    print("\n----------------正在绘制季节性差分图----------------\n")
    plt.figure(figsize=(10, 6))  # 设置图形大小
    df_total['季节性差分'].plot(kind='line', marker='o', color='blue', linestyle='-')  # 绘制折线图，添加颜色和线型
    plt.title(f"{need_to_predicted} - 季节性差分", fontsize=14, fontweight='bold')  # 设置标题，增加字体大小和加粗
    plt.xlabel('时间', fontsize=12)  # 添加X轴标签
    plt.ylabel('数值（亿元人民币）', fontsize=12)  # 添加Y轴标签
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，设置样式和透明度
    fig_path = os.path.join(figs_folder, f"深圳海关{need_to_predicted}_seasonal_diff.png")
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")
    plt.show()  # 显示图形
    print("\n----------------季节性差分图绘制完成----------------\n")

    # 绘制季节性差分ACF PACF图
    print("\n----------------正在绘制季节性差分ACF PACF图----------------\n")
    fig1 = plt.figure(figsize=(12, 8))  # 设置图形大小
    ax1 = fig1.add_subplot(211)  # 添加子图1
    fig1 = plot_acf(df_total['季节性差分'].iloc[13:], lags=30, ax=ax1, color='blue')  # 绘制自相关函数（ACF）图，设置颜色
    ax1.set_title('自相关函数（ACF）图', fontsize=12)  # 设置子图1标题
    ax1.set_xlabel('滞后阶数', fontsize=10)  # 设置X轴标签
    ax1.set_ylabel('自相关系数', fontsize=10)  # 设置Y轴标签

    ax2 = fig1.add_subplot(212)  # 添加子图2
    fig1 = plot_pacf(df_total['季节性差分'].iloc[13:], lags=20, ax=ax2, color='red')  # 绘制偏自相关函数（PACF）图，设置颜色
    ax2.set_title('偏自相关函数（PACF）图', fontsize=12)  # 设置子图2标题
    ax2.set_xlabel('滞后阶数', fontsize=10)  # 设置X轴标签
    ax2.set_ylabel('偏自相关系数', fontsize=10)  # 设置Y轴标签
    plt.tight_layout()  # 自动调整子图布局
    fig_path = os.path.join(figs_folder, f"深圳海关{need_to_predicted}_ACFandPACF.png")
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")
    plt.show()  # 显示图形
    print("\n----------------季节性差分ACF PACF图绘制完成----------------\n")


## SARIMA 模型参数
# 寻找最佳SARIMA模型参数
def find_best_sarima_params(train_data, max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2, s=12):
    best_aic = float("inf")
    best_bic = float("inf")
    best_order = None
    best_seasonal_order = None

    # 用于存储所有尝试的参数组合及其AIC和BIC值
    results = []

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', category=UserWarning)
                                try:
                                    model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
                                    model_fit = model.fit(disp=False)
                                    current_aic = model_fit.aic
                                    current_bic = model_fit.bic

                                    # 更新最佳AIC和BIC值
                                    if current_aic < best_aic:
                                        best_aic = current_aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, s)
                                    if current_bic < best_bic:
                                        best_bic = current_bic
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, s)

                                    # 打印当前结果
                                    print(f"order = ({p, d, q}), seasonal_order = ({P, D, Q, s})", end="\t")
                                    print(f'AIC: {current_aic}\tBIC: {current_bic}')

                                    # 将当前结果存储到列表中
                                    results.append({
                                        'order': (p, d, q),
                                        'seasonal_order': (P, D, Q, s),
                                        'AIC': current_aic,
                                        'BIC': current_bic
                                    })

                                except Exception as e:
                                    print(f"Error occurred for order=({p, d, q}) and seasonal_order=({P, D, Q, s}): {e}")
                                    continue

    # 将结果列表转换为DataFrame
    results_df = pd.DataFrame(results)
    # 保存文件内容
    excel_path = os.path.join(excels_folder, f"{need_to_predicted}_sarima_results.xlsx")
    results_df.to_excel(excel_path, index=True)
    print(f"参数组合及其AIC和BIC值已保存到文件：{excel_path}")

    # 打印最佳参数组合
    print(f'最佳ARIMA参数: {best_order}')
    print(f'最佳SARIMA参数: {best_seasonal_order}')
    print(f'AIC: {best_aic}, BIC: {best_bic}')

    return best_order, best_seasonal_order, best_aic, best_bic

# SARIMA 模型拟合
def fit_and_evaluate_sarima(test_time, test_data, history, order, seasonal_order):
    """
    拟合 SARIMA 模型，并对测试集进行预测，计算评估指标。
    
    参数:
    - test_data: 测试数据
    - history: 历史数据
    - order: SARIMA 模型的 (p, d, q) 参数
    - seasonal_order: SARIMA 模型的季节性参数 (P, D, Q, s)
    
    返回:
    - predictions: 预测结果列表
    - mse: 均方误差
    - rmse: 均方根误差
    - mae: 平均绝对误差
    - mape: 平均绝对百分比误差
    """
    # 初始化预测结果列表
    predictions = list()

    # 输出最佳模型参数
    print(f"当前最佳 SARIMA 参数：order={order}, seasonal_order={seasonal_order}")
    print("正在进行测试集拟合......")
    # 对测试集中的每个时间点进行预测
    for t in range(len(test_data)):
        # 使用历史数据拟合 SARIMA 模型
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        output = model_fit.get_forecast(steps=1) # 使用 get_forecast 方法
        yhat = output.predicted_mean[0]
        predictions.append(yhat)
        obs = test_data[t]
        history.append(obs)
        print(f'测试预测值：{yhat:.0f}\t实际值：{obs[0]}')
        # 将测试预测值和实际值保存到 DataFrame
    results_df = pd.DataFrame({
        '时间': test_time,
        '实际值': test_data[0][0],
        '预测值': predictions
    })

    # 保存文件内容
    excel_path = os.path.join(excels_folder, f"深圳海关{need_to_predicted}_test_predictions.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"测试预测值和实际值已保存到文件：{excel_path}")

    # 计算预测结果的评估指标
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

    # 输出评估指标
    print(f'实际值与测试预测值之间的均方误差 (MSE): {mse:.3f}')
    print(f'实际值与测试预测值之间的均方根误差 (RMSE): {rmse:.3f}')
    print(f'实际值与测试预测值之间的平均绝对误差 (MAE): {mae:.3f}')
    print(f'实际值与测试预测值之间的平均绝对百分比误差 (MAPE): {mape:.3f}%')
    
    return predictions, mse, rmse, mae, mape, model_fit


## 绘制函数
# 绘制连线图、箱线图
def plots_lineAndboxplot(df_total, titles):
    print("\n-----------------绘制图像内容-----------------\n")
    # 设置画布大小和布局
    plt.figure(figsize=(10, 4))
    plt.subplots_adjust(wspace=0.3)  # 子图之间的水平间距
    # 绘制当前数据连线图
    plt.subplot(1, 2, 1)  # 1行2列的第1个位置
    plt.plot(df_total.index, df_total[need_to_predicted], marker='o', color='blue', alpha=0.7)
    plt.title(f"{need_to_predicted} - {titles}数据连线图", fontsize=12)
    plt.xlabel('时间', fontsize=12)  # 假设 x 轴是日期
    plt.ylabel('数值（亿元人民币）', fontsize=10)
    # 绘制当前数据箱线图
    plt.subplot(1, 2, 2)  # 1行2列的第2个位置
    sns.boxplot(data=df_total[[need_to_predicted]], palette='viridis')
    plt.title(f"{need_to_predicted} - {titles}数据箱线图", fontsize=12)
    plt.ylabel("数值（亿元人民币）", fontsize=10)
    plt.tight_layout()
    # 保存图像
    fig_path = os.path.join(figs_folder, f"深圳海关{need_to_predicted}_{titles}_lineAndboxplot.png")
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")
    plt.show()
    print(f"\n-----------------{titles}图像绘制结束-----------------\n")

# 绘制原始数据与测试集预测数据对比图像
def plot_original_vs_predicted(df_total, test, predictions, need_to_predicted):
    print("\n-----------------绘制原始数据与预测数据对比图像-----------------\n")
    # 获取原始数据的索引和值
    original_index = df_total.index
    original_values = df_total[need_to_predicted]
    # 获取测试集的索引范围
    test_index = original_index[-len(test):]
    # 绘制图像
    plt.figure(figsize=(12, 6))
    plt.plot(original_index, original_values, label="原始数据", color="blue", marker="o", linestyle="-", linewidth=1.5, markersize=4)
    plt.plot(test_index, predictions, label="预测数据", color="red", marker="x", linestyle="-", linewidth=1.5, markersize=4)
    # 添加图例
    plt.legend(loc="best", fontsize=10)
    # 添加标题和轴标签
    plt.title(f"{need_to_predicted} - 原始数据与测试预测数据对比", fontsize=14, fontweight="bold")
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("数值（亿元人民币）", fontsize=12)
    # 添加网格线
    plt.grid(True, linestyle="--", alpha=0.7)
    # 显示图像
    plt.tight_layout()
    # 保存图像
    fig_path = os.path.join(figs_folder, f"深圳海关{need_to_predicted}_original_vs_predicted.png")
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")
    plt.show()
    print("\n-----------------对比图像绘制完成-----------------\n")

# 绘制原始数据与未来预测数据图像
def plot_future_predictions(df_total, model_fit, need_to_predicted, years_to_predicted):
    """
    绘制未来预测数据图像。
    
    参数:
    - df_total: 包含时间序列数据的 DataFrame
    - history: 历史数据列表
    - model_fit: 已拟合的 SARIMA 模型
    - need_to_predicted: 需要预测的列名
    - years_to_predicted: 需要预测的未来年数
    """
    print("\n-----------------绘制未来预测数据图像-----------------\n")
    # 获取原始数据的索引和值
    original_index = df_total.index
    original_values = df_total[need_to_predicted]
    # 计算未来预测的时间点
    future_steps = years_to_predicted * 12
    future_dates = pd.date_range(start=original_index[-1], periods=future_steps + 1, freq='MS')[1:]
    # 使用拟合好的模型进行未来预测
    future_predictions = model_fit.get_forecast(steps=future_steps).predicted_mean

    # 绘制图像
    plt.figure(figsize=(12, 6))
    plt.plot(original_index, original_values, label="原始数据", color="blue", marker="o", linestyle="-", linewidth=1.5, markersize=4)
    plt.plot(future_dates, future_predictions, label="未来预测", color="green", marker="x", linestyle="-", linewidth=1.5, markersize=4)

    # 添加图例
    plt.legend(loc="best", fontsize=10)
    # 添加标题和轴标签
    plt.title(f"{need_to_predicted} - 原始数据与未来预测数据对比", fontsize=14, fontweight="bold")
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("数值（亿元人民币）", fontsize=12)
    # 添加网格线
    plt.grid(True, linestyle="--", alpha=0.7)
    # 显示图像
    plt.tight_layout()
    # 保存图像
    fig_path = os.path.join(figs_folder, f"深圳海关{need_to_predicted}_future{years_to_predicted}years_predictions.png")
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")
    plt.show()
    print("\n-----------------未来预测数据图像绘制完成-----------------\n")


# 读取数据
df=pd.read_excel('深圳海关进出口数据2014~2024.xlsx')
# 进出口总额（亿元人民币） | 进口（亿元人民币） | 出口（亿元人民币）
need_to_predicted = str(input("请输入需要预测的内容|进出口总额（亿元人民币）|进口（亿元人民币）|出口（亿元人民币）|："))
years_to_predicted = int(input("请输入需要预测未来多少年的内容："))
df_total = df[['时间', need_to_predicted]].copy() # 保留单一列数据，用于预测。这里以进出口总额的预测为例。
df_total.set_index('时间',inplace=True) # 设置时间索引
print('原数据预览：\n', df.head())
print('保留后数据预览：\n', df_total.head(), end='\n-----------------------\n')

## 划分训练和测试集
X = df_total.values
if need_to_predicted != "进出口总额（亿元人民币）":
    trans = 0.7 # 根据数据特征划分
else:
    trans = 0.6
size = int(len(X) * trans)
train, test = X[0:size], X[size:len(X)]
test_time = df_total.index[size:len(X)] # 测试集时间索引
# 输出划分后的训练集和测试集
# print("训练集：")
# print(train)
# print("\n测试集：")
# print(test)
# 初始化历史数据列表，用于存储训练集中的数据
history = [x for x in train]

## 绘制原始数据连线图
plots_lineAndboxplot(df_total, titles="原始")

## SARIMA 参数
# best_order, best_seasonal_order, best_aic, best_bic = find_best_sarima_params(train) # 调用一次函数即可
"""
# (1, 0, 0) (2, 0, 1, 12) 总额
# (2, 0, 1) (2, 0, 1, 12) 进口
# (0, 0, 1) (1, 0, 0, 12) 出口
"""
best_order, best_seasonal_order = (1, 0, 0), (2, 0, 1, 12)


## SARIMA 拟合
predictions, mse, rmse, mae, mape, models = fit_and_evaluate_sarima(test_time, test, 
history, best_order, best_seasonal_order)


## ADF 检验
# 提取 ARIMA 和 SARIMA 的差分阶数
d = best_order[1]  # 非季节性差分阶数
D = best_seasonal_order[1]  # 季节性差分阶数
ADFs(d, D, df_total)


# 调用函数绘制对比图像
plot_original_vs_predicted(df_total, test, predictions, need_to_predicted)

# 调用函数绘制未来预测数据图像
plot_future_predictions(df_total, models, need_to_predicted, years_to_predicted)


# 获取未来预测的时间点和预测值
future_steps = years_to_predicted * 12
future_dates = pd.date_range(start=df_total.index[-1], periods=future_steps + 1, freq='MS')[1:]
future_predictions = models.get_forecast(steps=future_steps).predicted_mean

# 将未来预测数据添加到 df_total 中
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=[need_to_predicted])
df_total_with_predictions = pd.concat([df_total, future_df])

# 保存预测结果文件
excel_path = os.path.join(excels_folder, f"深圳海关{need_to_predicted}_{years_to_predicted}years_predicted_data.xlsx")
df_total_with_predictions.to_excel(excel_path, index=True)
print(f"数据已保存到文件：{excel_path}")