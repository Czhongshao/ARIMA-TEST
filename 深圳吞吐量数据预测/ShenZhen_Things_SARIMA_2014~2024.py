"""
# 深圳外贸货物吞吐量预测SARIMA模型

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
---
参数:
    series : pd.Series
        待检验的时间序列数据
    title : str, optional
        检验标题

返回值：
    控制台输出：
        - ADF检验统计量
        - p值
        - 临界值（1%/5%/10%）
        - 平稳性判定结论
    """

    # 执行ADF检验（返回统计量、p值、滞后阶数、样本数、临界值等）
    result = adfuller(series)

    # 格式化输出结果
    print(f"\nADF检验结果 - {title}:")
    print(f"检验统计量: {result[0]:.4f}")
    print(f"p值: {result[1]:.4f}")
    print("临界值:")
    for level, value in result[4].items():  # 遍历临界值字典
        print(f"   {level}: {value:.3f}")

    # 基于显著性水平0.05的平稳性判定
    print("\n结论: 序列" + ("平稳" if result[1] < 0.05 else "非平稳"))


def ADFs(d, D, df_total):
    """
    执行多阶差分ADF检验及可视化分析
---
参数:
    d : int
        非季节性差分阶数
    D : int
        季节性差分阶数
    df_total : pd.DataFrame
        包含待分析时间序列的DataFrame

返回值:
    控制台输出：
        - 各阶差分ADF检验结果
        - 季节性差分可视化图表
        - ACF/PACF分析图表
    文件保存：
        - 季节性差分时序图
        - ACF/PACF分析图
    """

    # === 1. 差分阶数检验 ===
    # 原始数据检验
    adfuller_test(df_total[need_to_predicted], title="原始数据")

    # 非季节性差分检验
    if d != 0:
        df_total[f'{d}阶差分'] = df_total[need_to_predicted].diff(d)  # d阶差分计算
        adfuller_test(df_total[f'{d}阶差分'].dropna(), title=f'{d}阶差分')

    # 季节性差分检验
    if D != 0:
        df_total[f'{D}阶季节性差分'] = df_total[need_to_predicted].diff(D)
        adfuller_test(df_total[f'{D}阶季节性差分'].dropna(), title=f'{D}阶季节性差分')

    # === 2. 季节性分析 ===
    # 固定12期季节性差分（货运量典型周期）
    df_total['季节性差分'] = df_total[need_to_predicted].diff(12)

    # 2.1 季节性差分检验
    adfuller_test(df_total['季节性差分'].dropna(), title='12期季节性差分')

    # 2.2 时序图可视化
    print("\n" + "=" * 20 + " 季节性差分时序图 " + "=" * 20)
    plt.figure(figsize=(10, 6))
    df_total['季节性差分'].plot(
        kind='line',
        marker='o',
        color='blue',
        linestyle='-',
        title="深圳货物吞吐量季节性差分",
        xlabel='时间',
        ylabel='差值（万吨）'
    )
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("季节性差分时序图.png")
    plt.show()

    # === 3. 自相关分析 ===
    print("\n" + "=" * 20 + " ACF/PACF分析 " + "=" * 20)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 3.1 ACF图（30阶滞后）
    plot_acf(df_total['季节性差分'].iloc[13:],
             lags=30,
             ax=ax1,
             color='blue',
             title="自相关函数(ACF)")

    # 3.2 PACF图（20阶滞后）
    plot_pacf(df_total['季节性差分'].iloc[13:],
              lags=20,
              ax=ax2,
              color='red',
              title="偏自相关函数(PACF)")

    plt.tight_layout()
    plt.savefig("ACF_PACF分析.png")
    plt.show()


def find_best_sarima_params(train_data, max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2, s=12):
    """
    SARIMA模型参数网格搜索
    ---
    功能：
        通过网格搜索寻找最优SARIMA(p,d,q)(P,D,Q,s)参数组合

    参数：
        train_data : pd.Series
            训练集时间序列数据
        max_p/max_q/max_d : int, optional
            非季节性AR/I/MA最大阶数（默认0-5）
        max_P/max_Q/max_D : int, optional
            季节性AR/I/MA最大阶数（默认0-2）
        s : int, optional
            季节周期（默认12）

    返回：
        best_order : tuple
            最优非季节性阶数(p,d,q)
        best_seasonal_order : tuple
            最优季节性阶数(P,D,Q,s)
        best_aic : float
            最小AIC值
        best_bic : float
            最小BIC值

    输出：
        1. 控制台实时打印各参数组合的AIC/BIC
        2. 保存完整参数组合结果到Excel
        3. 输出最优参数组合
    """

    # 初始化最优指标
    best_aic = float("inf")
    best_bic = float("inf")
    best_order = None
    best_seasonal_order = None
    results = []  # 存储所有参数组合结果

    # === 参数网格搜索 ===
    for p in range(max_p + 1):  # AR阶数
        for d in range(max_d + 1):  # 差分阶数
            for q in range(max_q + 1):  # MA阶数
                for P in range(max_P + 1):  # 季节性AR
                    for D in range(max_D + 1):  # 季节性差分
                        for Q in range(max_Q + 1):  # 季节性MA

                            # 忽略模型拟合过程中的警告
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')

                                try:
                                    # 1. 模型初始化与拟合
                                    model = SARIMAX(
                                        train_data,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, s)
                                    )
                                    model_fit = model.fit(disp=False)

                                    # 2. 获取评估指标
                                    current_aic = model_fit.aic
                                    current_bic = model_fit.bic

                                    # 3. 更新最优参数
                                    if current_aic < best_aic:
                                        best_aic, best_order, best_seasonal_order = current_aic, (p, d, q), (P, D, Q, s)
                                    if current_bic < best_bic:
                                        best_bic = current_bic

                                    # 4. 打印当前结果
                                    print(f"SARIMA({p},{d},{q})({P},{D},{Q})_{s}",
                                          f"AIC:{current_aic:.1f}",
                                          f"BIC:{current_bic:.1f}", sep=' | ')

                                    # 5. 存储结果
                                    results.append({
                                        'order': (p, d, q),
                                        'seasonal_order': (P, D, Q, s),
                                        'AIC': current_aic,
                                        'BIC': current_bic
                                    })

                                except Exception as e:
                                    print(f"参数组合({p},{d},{q})({P},{D},{Q})失败: {str(e)[:50]}...")
                                    continue

    # === 结果保存与输出 ===
    # 1. 保存完整结果
    results_df = pd.DataFrame(results).sort_values('AIC')
    excel_path = os.path.join(excels_folder, "SARIMA参数搜索.xlsx")
    results_df.to_excel(excel_path)

    # 2. 输出最优结果
    print(f"\n{' 最优参数 ':=^40}")
    print(f"非季节性阶数: {best_order}")
    print(f"季节性阶数: {best_seasonal_order}")
    print(f"AIC: {best_aic:.2f} | BIC: {best_bic:.2f}")

    return best_order, best_seasonal_order, best_aic, best_bic


def fit_and_evaluate_sarima(test_time, test_data, history, order, seasonal_order):
    """
    SARIMA模型滚动预测与评估
    ---
    功能：
        1. 使用历史数据滚动拟合SARIMA模型
        2. 对测试集进行逐步预测
        3. 计算并输出四大评估指标
        4. 保存预测结果

    参数：
        test_time : array-like
            测试集时间索引
        test_data : array-like
            测试集实际值
        history : list
            初始训练数据（动态更新）
        order : tuple
            非季节性参数(p,d,q)
        seasonal_order : tuple
            季节性参数(P,D,Q,s)

    返回：
        predictions : list
            测试集预测值序列
        mse/rmse/mae/mape : float
            评估指标
        model_fit : statsmodels模型对象
            最终训练完成的模型

    过程：
        1. 逐步将测试点加入历史数据
        2. 每次重新拟合模型
        3. 计算单步预测值
    """
    predictions = list()  # 存储预测结果

    # 打印模型参数
    print(f"当前最佳 SARIMA 参数：order={order}, seasonal_order={seasonal_order}")
    print("正在进行测试集拟合......")

    # 滚动预测流程
    for t in range(len(test_data)):
        # 模型拟合（禁用稳定性强制约束）
        model = SARIMAX(history,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)

        # 单步预测
        output = model_fit.get_forecast(steps=1)
        yhat = output.predicted_mean[0]
        predictions.append(yhat)

        # 更新历史数据
        obs = test_data[t]
        history.append(obs)

        # 打印当前预测结果
        print(f'测试预测值：{yhat:.0f}\t实际值：{obs[0]}')

    # 保存预测结果
    results_df = pd.DataFrame({
        '时间': test_time,
        '实际值': test_data[0][0],
        '预测值': predictions
    })
    excel_path = os.path.join(excels_folder, "深圳外贸货物吞吐量_测试集预测结果.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"测试预测值和实际值已保存到文件：{excel_path}")

    # 计算评估指标
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

    # 打印评估报告
    print('\n[模型评估结果]')
    print(f'MSE:  {mse:.3f}')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE:  {mae:.3f}')
    print(f'MAPE: {mape:.3f}%')

    return predictions, mse, rmse, mae, mape, model_fit


def plots_lineAndboxplot(df_total, titles):
    """
    绘制时间序列折线图与箱线图组合
    ---
    功能：
        1. 绘制时间序列折线图（含数据点标记）
        2. 绘制数值分布箱线图
        3. 自动保存组合图表

    参数：
        df_total : pd.DataFrame
            包含时间序列数据的DataFrame
        titles : str
            图表标题后缀（如"原始数据"、"差分数据"等）

    输出：
        1. 显示组合图表
        2. 保存PNG格式图片到指定路径
        3. 控制台输出保存路径
    """

    # === 1. 初始化设置 ===
    print("\n" + "=" * 20 + f" 开始绘制[{titles}]图表 " + "=" * 20)
    plt.figure(figsize=(10, 4))
    plt.subplots_adjust(wspace=0.3)  # 设置子图水平间距

    # === 2. 绘制折线图 ===
    plt.subplot(1, 2, 1)  # 创建左子图
    plt.plot(
        df_total.index,  # x轴数据（时间索引）
        df_total[need_to_predicted],  # y轴数据
        marker='o',
        color='blue',
        alpha=0.7
    )
    plt.title(f"{titles} - 时间序列", fontsize=12)
    plt.xlabel('时间', fontsize=10)
    plt.ylabel('吞吐量（万吨）', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    # === 3. 绘制箱线图 ===
    plt.subplot(1, 2, 2)  # 创建右子图
    sns.boxplot(
        data=df_total[[need_to_predicted]],
        palette='viridis',
        width=0.5
    )
    plt.title(f"{titles} - 数据分布", fontsize=12)
    plt.ylabel("")
    plt.xticks([])

    # === 4. 保存与输出 ===
    plt.tight_layout()  # 自动调整子图间距
    fig_path = os.path.join(figs_folder, f"吞吐量_{titles}_分析图.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至：\n{fig_path}")
    plt.show()

    print("\n" + "=" * 20 + f" [{titles}]图表绘制完成 " + "=" * 20)


def plot_original_vs_predicted(df_total, test, predictions, need_to_predicted):
    """
    绘制原始数据与预测数据对比图
    ---
    功能：
        1. 可视化展示完整时间序列与测试集预测结果的对比
        2. 突出显示预测区间
        3. 自动保存高清对比图

    参数：
        df_total : pd.DataFrame
            包含完整时间序列的DataFrame
        test : array-like
            测试集实际值数据
        predictions : array-like
            测试集预测值数据
        need_to_predicted : str
            目标列名

    输出：
        1. 显示带标记点的对比折线图
        2. 保存PNG格式图片到指定路径
        3. 控制台输出保存状态
    """

    # === 1. 数据准备 ===
    print("\n" + "=" * 40 + "\n 正在生成预测对比图 \n" + "=" * 40)
    original_index = df_total.index
    original_values = df_total[need_to_predicted]
    test_index = original_index[-len(test):]

    # === 2. 绘图设置 ===
    plt.figure(figsize=(12, 6))

    # 2.1 绘制原始数据（蓝色实线+圆点标记）
    plt.plot(original_index, original_values,
             label="原始数据",
             color="blue",
             marker="o",
             linestyle="-",
             linewidth=1.5,
             markersize=4,
             alpha=0.8)

    # 2.2 绘制预测数据（红色实线+X形标记）
    plt.plot(test_index, predictions,
             label="预测数据",
             color="red",
             marker="x",
             linestyle="-",
             linewidth=1.5,
             markersize=6,
             alpha=0.9)

    # === 3. 图表美化 ===
    plt.legend(loc="upper left", fontsize=10)  # 左上角图例
    plt.title("深圳外贸货物吞吐量 - 实际 vs 预测", fontsize=14, pad=20)
    plt.xlabel("时间", fontsize=12, labelpad=10)
    plt.ylabel("吞吐量 (万吨)", fontsize=12, labelpad=10)

    # 3.1 网格线设置
    plt.grid(True,
             linestyle="--",
             alpha=0.5,
             which='both')

    # 3.2 突出显示预测区间
    plt.axvspan(test_index[0], test_index[-1],
                facecolor='yellow',
                alpha=0.1)

    # === 4. 输出保存 ===
    plt.tight_layout()
    fig_path = os.path.join(figs_folder, "吞吐量_实际预测对比.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至：{fig_path}")
    plt.show()
    print("\n" + "=" * 40 + "\n 对比图生成完成 \n" + "=" * 40)


def plot_future_predictions(df_total, model_fit, need_to_predicted, years_to_predicted):
    """
    未来预测可视化（保持原代码不变）
    ---
    功能：
        1. 绘制历史数据与未来预测的对比曲线
        2. 自动生成未来时间序列（月度）
        3. 保存高清对比图表

    参数：
        df_total : pd.DataFrame
            含历史数据的DataFrame（需含时间索引）
        model_fit : SARIMAXResults
            已完成训练的SARIMA模型
        need_to_predicted : str
            目标变量列名
        years_to_predicted : int
            预测年数（自动转为月份数）

    输出：
        1. 显示带标记点的对比折线图
        2. 保存PNG格式图片
        3. 控制台输出保存路径
    """

    print("\n-----------------绘制未来预测数据图像-----------------\n")
    original_index = df_total.index
    original_values = df_total[need_to_predicted]
    # 计算未来预测的时间点
    future_steps = years_to_predicted * 12
    future_dates = pd.date_range(start=original_index[-1], periods=future_steps + 1, freq='MS')[1:]
    # 使用拟合好的模型进行未来预测
    future_predictions = model_fit.get_forecast(steps=future_steps).predicted_mean

    plt.figure(figsize=(12, 6))
    plt.plot(original_index, original_values, label="原始数据", color="blue", marker="o", linestyle="-", linewidth=1.5, markersize=4)
    plt.plot(future_dates, future_predictions, label="未来预测", color="green", marker="x", linestyle="-", linewidth=1.5, markersize=4)

    plt.legend(loc="best", fontsize=10)
    plt.title(f"深圳货物吞吐量 - 原始数据与未来预测数据对比", fontsize=14, fontweight="bold")
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("数值（万吨）", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    # 保存图像
    fig_path = os.path.join(figs_folder, f"深圳货物吞吐量_future{years_to_predicted}years_predictions.png")
    plt.savefig(fig_path)
    print(f"图像已保存到文件：{fig_path}")
    plt.show()
    print("\n-----------------未来预测数据图像绘制完成-----------------\n")


df=pd.read_excel('深圳外贸货物吞吐量.xlsx')
need_to_predicted = "外贸货物吞吐量(万吨)"
years_to_predicted = int(input("请输入需要预测未来多少年的内容："))
df_total = df[['时间', need_to_predicted]].copy() # 保留单一列数据，用于预测。这里以进出口总额的预测为例。
df_total.set_index('时间',inplace=True) # 设置时间索引
print('原数据预览：\n', df.head())
print('保留后数据预览：\n', df_total.head(), end='\n-----------------------\n')

## 划分训练和测试集
X = df_total.values
size = int(len(X) * 0.75)
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
最佳ARIMA参数: (0, 1, 1)
最佳SARIMA参数: (1, 1, 2, 12)
"""
best_order, best_seasonal_order = (0, 1, 1), (1, 1, 2, 12)


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
excel_path = os.path.join(excels_folder, f"深圳货物吞吐量_{years_to_predicted}years_predicted_data.xlsx")
df_total_with_predictions.to_excel(excel_path, index=True)
print(f"数据已保存到文件：{excel_path}")