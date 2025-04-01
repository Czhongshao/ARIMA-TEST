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


def adfuller_test(series, title=""):
    """
    Augmented Dickey-Fuller单位根检验
    ----------------------------
    功能: 检验时间序列的平稳性，判断是否存在单位根

    参数:
        series : pd.Series/array-like
            待检验的时间序列数据
        title : str, optional
            检验标题(用于结果标识)

    输出:
        打印检验结果和统计结论
    """

    # ==== 1. 检验执行 ====
    print(f"\nADF 检验 - {title}")

    # 调用statsmodels的adfuller函数
    # 返回值: (统计量, p值, 滞后阶数, 样本量, 临界值字典, 其他信息)
    result = adfuller(series)

    # ==== 2. 结果输出 ====
    # -- 核心指标 --
    labels = ['ADF统计量', 'p值', '滞后阶数', '样本量']
    for value, label in zip(result[:4], labels):  # 只输出前四个核心结果
        print(f"{label}: {value}")

    # -- 临界值 --
    print("临界值:")
    for level, value in result[4].items():  # 遍历临界值字典
        print(f"  {level}%: {value:.4f}")

    # ==== 3. 结论判断 ====
    print("\n结论:")
    if result[1] <= 0.05:
        print("-> 序列平稳(p值={:.4f}≤0.05，拒绝原假设，不存在单位根)".format(result[1]))
    else:
        print("-> 序列非平稳(p值={:.4f}>0.05，无法拒绝原假设，存在单位根)".format(result[1]))

def ADFs(d, D, df_total):
    """
    差分数据ADF检验与可视化分析
    --------------------------
    功能:
    1. 对原始数据及差分后数据进行ADF平稳性检验
    2. 绘制季节性差分时序图
    3. 绘制ACF/PACF自相关图

    参数:
    d : int
        非季节性差分阶数
    D : int
        季节性差分阶数
    df_total : DataFrame
        包含待分析时间序列的数据框
    """

    # ==== 1. ADF平稳性检验 ====
    print("\n" + "=" * 50)
    print("开始ADF平稳性检验...")

    # -- 1.1 原始数据检验 --
    adfuller_test(df_total[need_to_predicted], title="原始数据")

    # -- 1.2 非季节差分检验 --
    if d != 0:
        df_total[f'{d}阶差分'] = df_total[need_to_predicted].diff(d)  # d阶差分
        adfuller_test(df_total[f'{d}阶差分'].dropna(), title=f'{d}阶差分')

    # -- 1.3 季节差分检验 --
    if D != 0:
        df_total[f'{D}阶季节差分'] = df_total[need_to_predicted].diff(D)
        adfuller_test(df_total[f'{D}阶季节差分'].dropna(), title=f'{D}阶季节差分')

    # 固定12期季节差分(用于后续分析)
    df_total['季节性差分'] = df_total[need_to_predicted].diff(12)
    adfuller_test(df_total['季节性差分'].dropna(), title='12期季节差分')

    # ==== 2. 数据可视化 ====
    print("\n" + "=" * 50)
    print("开始数据可视化...")

    # -- 2.1 季节差分时序图 --
    plt.figure(figsize=(10, 6))
    df_total['季节性差分'].plot(
        kind='line',
        marker='o',
        color='blue',
        linestyle='-'
    )
    plt.title(f"{need_to_predicted} - 季节性差分", fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('差分值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    fig_path = os.path.join(figs_folder, f"seasonal_diff.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"-> 季节差分图已保存: {fig_path}")

    # -- 2.2 ACF/PACF图 --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # ACF图
    plot_acf(df_total['季节性差分'].dropna(),
             lags=30,
             ax=ax1,
             color='blue',
             title='自相关函数(ACF)')

    # PACF图
    plot_pacf(df_total['季节性差分'].dropna(),
              lags=20,
              ax=ax2,
              color='red',
              title='偏自相关函数(PACF)')

    plt.tight_layout()
    fig_path = os.path.join(figs_folder, f"ACF_PACF.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"-> ACF/PACF图已保存: {fig_path}")

    print("\n" + "=" * 50)
    print("所有分析完成!")

def find_best_sarima_params(train_data, max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2, s=12):
    """
    功能: 通过网格搜索寻找最优SARIMA参数组合
    参数:
        train_data: 输入时间序列数据
        max_p: AR项最大阶数
        max_d: 差分最大阶数
        max_q: MA项最大阶数
        max_P: 季节性AR最大阶数
        max_D: 季节性差分最大阶数
        max_Q: 季节性MA最大阶数
        s: 季节周期长度
    返回:
        最优参数组合及对应的AIC/BIC值
    """

    # ==== 1. 初始化变量 ====
    best_aic = float("inf")  # 最小AIC初始值
    best_bic = float("inf")  # 最小BIC初始值
    best_order = None  # 最优(p,d,q)组合
    best_seasonal_order = None  # 最优(P,D,Q,s)组合
    results = []  # 存储所有参数组合结果

    # ==== 2. 参数网格搜索 ====
    for p in range(max_p + 1):  # 遍历AR阶数
        for d in range(max_d + 1):  # 遍历差分阶数
            for q in range(max_q + 1):  # 遍历MA阶数
                for P in range(max_P + 1):  # 遍历季节AR
                    for D in range(max_D + 1):  # 遍历季节差分
                        for Q in range(max_Q + 1):  # 遍历季节MA

                            # -- 2.1 模型训练 --
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', category=UserWarning)
                                try:
                                    # 创建SARIMA模型
                                    model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
                                    # 模型拟合
                                    model_fit = model.fit(disp=False)

                                    # -- 2.2 指标计算 --
                                    current_aic = model_fit.aic
                                    current_bic = model_fit.bic

                                    # -- 2.3 更新最优值 --
                                    if current_aic < best_aic:
                                        best_aic = current_aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, s)
                                    if current_bic < best_bic:
                                        best_bic = current_bic
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, s)

                                    # -- 2.4 记录结果 --
                                    print(f"order = ({p, d, q}), seasonal_order = ({P, D, Q, s})", end="\t")
                                    print(f'AIC: {current_aic}\tBIC: {current_bic}')
                                    results.append({
                                        'order': (p, d, q),
                                        'seasonal_order': (P, D, Q, s),
                                        'AIC': current_aic,
                                        'BIC': current_bic
                                    })

                                except Exception as e:
                                    print(
                                        f"Error occurred for order=({p, d, q}) and seasonal_order=({P, D, Q, s}): {e}")
                                    continue

    # ==== 3. 结果输出 ====
    # 3.1 保存结果到Excel
    results_df = pd.DataFrame(results)
    excel_path = os.path.join(excels_folder, f"{need_to_predicted}_sarima_results.xlsx")
    results_df.to_excel(excel_path, index=True)
    print(f"参数组合及其AIC和BIC值已保存到文件：{excel_path}")

    # 3.2 打印最优参数
    print(f'最佳ARIMA参数: {best_order}')
    print(f'最佳SARIMA参数: {best_seasonal_order}')
    print(f'AIC: {best_aic}, BIC: {best_bic}')

    return best_order, best_seasonal_order, best_aic, best_bic


def fit_and_evaluate_sarima(test_time, test_data, history, order, seasonal_order):
    """
    SARIMA模型预测评估流程
    ---------------------
    1. 使用历史数据滚动拟合SARIMA模型
    2. 对测试集进行逐步预测
    3. 计算并返回预测评估指标

    参数:
    test_time : array-like
        测试集时间索引
    test_data : array-like
        测试集实际值(形状需与history一致)
    history : list
        历史训练数据(动态扩展)
    order : tuple
        (p,d,q)非季节参数
    seasonal_order : tuple
        (P,D,Q,s)季节参数

    返回:
    tuple: (预测值数组, MSE, RMSE, MAE, MAPE, 模型对象)
    """

    # ==== 1. 初始化 ====
    predictions = []  # 预测结果容器
    print(f"模型参数 order={order} seasonal_order={seasonal_order}")

    # ==== 2. 滚动预测 ====
    print("开始测试集预测...")
    for t in range(len(test_data)):
        # -- 2.1 模型拟合 --
        model = SARIMAX(history,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,  # 不强制平稳
                        enforce_invertibility=False)  # 不强制可逆
        model_fit = model.fit(disp=False)  # 不显示优化过程

        # -- 2.2 单步预测 --
        forecast = model_fit.get_forecast(steps=1)
        yhat = forecast.predicted_mean[0]  # 提取预测值
        predictions.append(yhat)

        # -- 2.3 更新历史 --
        history.append(test_data[t])  # 将真实值加入历史
        print(f"[step {t + 1}] 预测: {yhat:.1f} | 实际: {test_data[t][0]:.1f}")

    # ==== 3. 结果保存 ====
    results_df = pd.DataFrame({
        '时间': test_time,
        '实际值': test_data[:, 0],  # 假设test_data是二维数组
        '预测值': predictions
    })
    excel_path = os.path.join(excels_folder,
                              f"{need_to_predicted}_test_predictions.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"预测结果已保存至 {excel_path}")

    # ==== 4. 评估计算 ====
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    mape = 100 * np.mean(np.abs((test_data - predictions) / test_data))

    print("\n评估指标:")
    print(f"MSE: {mse:.3f}  RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}  MAPE: {mape:.1f}%")

    return predictions, mse, rmse, mae, mape, model_fit


def plots_lineAndboxplot(df_total, titles):
    """
    绘制时间序列的折线图和箱线图
    --------------------------
    参数:
        df_total: 包含时间序列数据的DataFrame
        titles: 用于图表标题的字符串标识
    """
    # ==== 1. 初始化设置 ====
    print("\n-----------------绘制图像内容-----------------\n")

    # -- 1.1 创建画布 --
    plt.figure(figsize=(10, 4))
    plt.subplots_adjust(wspace=0.3)

    # ==== 2. 折线图绘制 ====
    plt.subplot(1, 2, 1)

    # -- 2.1 绘制折线 --
    plt.plot(df_total.index,
             df_total[need_to_predicted],
             marker='o',
             color='blue',
             alpha=0.7)

    # -- 2.2 图表装饰 --
    plt.title(f"{need_to_predicted} - {titles}数据连线图", fontsize=12)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('数值（亿元人民币）', fontsize=10)

    # ==== 3. 箱线图绘制 ====
    plt.subplot(1, 2, 2)

    # -- 3.1 绘制箱线 --
    sns.boxplot(data=df_total[[need_to_predicted]],
                palette='viridis')

    # -- 3.2 图表装饰 --
    plt.title(f"{need_to_predicted} - {titles}数据箱线图", fontsize=12)
    plt.ylabel("数值（亿元人民币）", fontsize=10)

    # ==== 4. 输出保存 ====
    plt.tight_layout()

    # -- 4.1 保存图片 --
    fig_path = os.path.join(figs_folder, f"深圳海关{need_to_predicted}_{titles}_lineAndboxplot.png")
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")

    # -- 4.2 显示图片 --
    plt.show()
    print(f"\n-----------------{titles}图像绘制结束-----------------\n")


def plot_original_vs_predicted(df_total, test, predictions, need_to_predicted):
    """
    绘制原始数据与预测数据对比图
    --------------------------
    参数:
        df_total: 包含完整时间序列的DataFrame
        test: 测试集数据
        predictions: 模型预测结果
        need_to_predicted: 目标变量名称
    """
    # ==== 1. 初始化设置 ====
    print("\n-----------------绘制原始数据与预测数据对比图像-----------------\n")

    # -- 1.1 准备数据 --
    original_index = df_total.index
    original_values = df_total[need_to_predicted]
    test_index = original_index[-len(test):]

    # ==== 2. 绘图设置 ====
    plt.figure(figsize=(12, 6))

    # -- 2.1 绘制原始数据 --
    plt.plot(original_index, original_values,
             label="原始数据",
             color="blue",
             marker="o",
             linestyle="-",
             linewidth=1.5,
             markersize=4)

    # -- 2.2 绘制预测数据 --
    plt.plot(test_index, predictions,
             label="预测数据",
             color="red",
             marker="x",
             linestyle="-",
             linewidth=1.5,
             markersize=4)

    # ==== 3. 图表装饰 ====
    # -- 3.1 图例设置 --
    plt.legend(loc="best", fontsize=10)

    # -- 3.2 标题和坐标轴 --
    plt.title(f"{need_to_predicted} - 原始数据与测试预测数据对比",
              fontsize=14,
              fontweight="bold")
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("数值（亿元人民币）", fontsize=12)

    # -- 3.3 网格线设置 --
    plt.grid(True, linestyle="--", alpha=0.7)

    # ==== 4. 输出保存 ====
    plt.tight_layout()

    # -- 4.1 保存图像 --
    fig_path = os.path.join(figs_folder,
                            f"深圳海关{need_to_predicted}_original_vs_predicted.png")
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")

    # -- 4.2 显示图像 --
    plt.show()
    print("\n-----------------对比图像绘制完成-----------------\n")


def plot_future_predictions(df_total, model_fit, need_to_predicted, years_to_predicted):
    """
    绘制时间序列未来预测结果
    ----------------------
    功能:
    1. 基于已拟合模型生成未来预测
    2. 可视化展示历史数据与预测结果对比

    参数:
        df_total: 包含历史数据的DataFrame
        model_fit: 已训练好的SARIMA模型
        need_to_predicted: 目标变量名称(str)
        years_to_predicted: 预测年数(int)
    """
    # ==== 1. 初始化设置 ====
    print("\n-----------------绘制未来预测数据图像-----------------\n")

    # -- 1.1 准备历史数据 --
    original_index = df_total.index
    original_values = df_total[need_to_predicted]

    # ==== 2. 生成预测 ====
    # -- 2.1 计算预测期数 --
    future_steps = years_to_predicted * 12

    # -- 2.2 生成未来日期 --
    future_dates = pd.date_range(
        start=original_index[-1],
        periods=future_steps + 1,
        freq='MS'
    )[1:]

    # -- 2.3 执行预测 --
    future_predictions = model_fit.get_forecast(
        steps=future_steps
    ).predicted_mean

    # ==== 3. 绘图展示 ====
    plt.figure(figsize=(12, 6))

    # -- 3.1 绘制历史数据 --
    plt.plot(
        original_index,
        original_values,
        label="原始数据",
        color="blue",
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=4
    )

    # -- 3.2 绘制预测数据 --
    plt.plot(
        future_dates,
        future_predictions,
        label="未来预测",
        color="green",
        marker="x",
        linestyle="-",
        linewidth=1.5,
        markersize=4
    )

    # ==== 4. 图表装饰 ====
    # -- 4.1 添加图例 --
    plt.legend(loc="best", fontsize=10)

    # -- 4.2 设置标题 --
    plt.title(
        f"{need_to_predicted} - 原始数据与未来预测数据对比",
        fontsize=14,
        fontweight="bold"
    )

    # -- 4.3 坐标轴设置 --
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("数值（亿元人民币）", fontsize=12)

    # -- 4.4 网格线设置 --
    plt.grid(True, linestyle="--", alpha=0.7)

    # ==== 5. 输出保存 ====
    plt.tight_layout()

    # -- 5.1 保存图像 --
    fig_path = os.path.join(
        figs_folder,
        f"深圳海关{need_to_predicted}_future{years_to_predicted}years_predictions.png"
    )
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")

    # -- 5.2 显示图像 --
    plt.show()
    print("\n-----------------未来预测数据图像绘制完成-----------------\n")

# 读取数据
df=pd.read_excel('深圳海关进出口数据2014~2024.xlsx')
# 进出口总额（亿元人民币） | 进口（亿元人民币） | 出口（亿元人民币）
need_to_predicted = str(input("请输入需要预测的内容[进出口总额（亿元人民币）|进口（亿元人民币）|出口（亿元人民币）]："))
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
# (1, 0, 0), (2, 0, 1, 12) 总额
# (2, 0, 1), (2, 0, 1, 12) 进口
# (0, 0, 1), (1, 0, 0, 12) 出口
"""
# 进出口总额（亿元人民币）|进口（亿元人民币）|出口（亿元人民币）|
if need_to_predicted == "进出口总额（亿元人民币）":
    best_order, best_seasonal_order = (1, 0, 0), (2, 0, 1, 12)
elif need_to_predicted == "进口（亿元人民币）":
    best_order, best_seasonal_order = (2, 0, 1), (2, 0, 1, 12)
elif need_to_predicted == "出口（亿元人民币）":
    best_order, best_seasonal_order = (0, 0, 1), (1, 0, 0, 12)


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