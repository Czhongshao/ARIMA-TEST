"""
# 深圳签订外贸合同量预测ARIMA模型

## 使用到的模块及其安装

 - Python version: 3.11.0

 - Pandas version: 2.2.3
    - pip install pandas==2.2.3

 - NumPy version: 2.2.2
    - pip install numpy==2.2.2

 - Matplotlib version: 3.10.1
    - pip install matplotlib==3.10.1

 - Seaborn version: 0.13.2
    - pip install seaborn==0.13.2
    
 - Statsmodels version:
    - pip install statsmodels==0.14.4

 - SciKit-Learn version: 1.6.1
    - pip install scikit-learn==1.6.1

 - TensorFlow version: 2.19.0
    - pip install tensorflow==2.19.0
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_excel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from matplotlib.dates import DateFormatter

# 设定Matplotlib字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def find_optimal_diff(series, max_diff=2):
    """自动寻找最优差分阶数
    Args:
        series: 待差分的时序数据
        max_diff: 最大尝试差分阶数（默认2）
    Returns:
        使序列平稳的最小差分阶数
    """
    best_d = 0
    current_series = series.copy()  # 保留原始数据副本

    # 从0阶开始逐阶测试差分
    for d in range(0, max_diff + 1):
        # 当前阶数差分结果（0阶直接使用原序列）
        test_series = current_series if d == 0 else difference(current_series, 1)

        # ADF检验p值判断平稳性
        p_value = adfuller(test_series.dropna())[1]

        if p_value < 0.05:  # 通过平稳性检验
            best_d = d
            break

        current_series = test_series  # 保留当前差分结果供下次迭代

    return best_d


def difference(series, d=1):
    """递归差分函数
    Args:
        series: 待差分的时序数据
        d: 差分阶数（默认1阶）
    Returns:
        完成d阶差分后的序列（自动去除NaN）
    """
    # 递归执行d次一阶差分
    for _ in range(d):
        series = series.diff().dropna()  # 单步差分+去空值
    return series


def inverse_difference(history, yhat_diff, d):
    """多阶逆差分（确保非负）
    Args:
        history: 历史原始数据（用于逆差分计算）
        yhat_diff: 差分后的预测值
        d: 差分阶数
    Returns:
        还原后的预测值（保证非负）
    """
    yhat = yhat_diff
    # 按差分阶数逐步还原
    for i in range(d):
        yhat += history[-(i + 1)]  # 逐阶加上历史值

    return max(yhat, 0)  # 结果截断至非负


def create_dataset(data, look_back=3):
    """创建时间序列数据集（滑动窗口方法）
    Args:
        data: 原始时间序列数据（1D数组）
        look_back: 输入序列长度/滑动窗口大小（默认3）
    Returns:
        X: 输入序列数组（形状：[n_samples, look_back]）
        Y: 输出值数组（形状：[n_samples]）
    """
    X, Y = [], []

    # 滑动窗口遍历数据
    for i in range(len(data) - look_back):
        # 取当前窗口作为输入特征
        X.append(data[i:(i + look_back)])
        # 取下一个时间点作为目标值
        Y.append(data[i + look_back])

    return np.array(X), np.array(Y)


def build_lstm_model(look_back, neurons=256):
    """改进的LSTM模型架构
    Args:
        look_back: 输入序列的时间步长
        neurons: 各LSTM层神经元数量（默认256）
    Returns:
        编译好的Keras Sequential模型
    """
    model = Sequential([
        # 第一层LSTM（返回完整序列供下一层使用）
        LSTM(neurons, return_sequences=True, input_shape=(1, look_back)),
        Dropout(0.3),  # 第一层Dropout

        # 第二层LSTM（继续返回序列）
        LSTM(neurons, return_sequences=True),
        Dropout(0.3),  # 第二层Dropout

        # 第三层LSTM（仅返回最后时间步）
        LSTM(neurons),
        Dropout(0.3),  # 第三层Dropout

        # 输出层（单值预测）
        Dense(1)
    ])

    # 编译配置（均方误差损失+Adam优化器）
    model.compile(loss='mse', optimizer='adam')
    return model


def plot_results(true, pred, future_pred, country):
    """增强的可视化函数（支持历史数据、测试预测和未来预测对比）
    Args:
        true: 完整真实值序列
        pred: 测试集预测结果
        future_pred: 未来多步预测结果
        country: 国家名称（用于标题和文件名）
    """
    plt.figure(figsize=(14, 7))  # 设置画布尺寸

    # 生成历史数据日期范围（年度数据）
    dates = pd.date_range(start='1990', periods=len(true), freq='YE')
    plt.plot(dates, true, label='实际值', color='blue', marker='o')  # 原始数据曲线

    # 测试集预测结果绘制
    test_dates = dates[-len(pred):]
    plt.plot(test_dates, pred, label='测试预测',
             color='red', linestyle='--', marker='x')  # 测试预测虚线

    # 未来预测结果绘制
    future_dates = pd.date_range(
        start=dates[-1] + pd.DateOffset(years=1),
        periods=len(future_pred),
        freq='YE'
    )
    plt.plot(future_dates, future_pred, label='未来预测',
             color='green', linestyle='--', marker='^')  # 未来预测虚线

    # 图表装饰元素
    plt.title(f'{country} - 外贸合同签订项数预测', fontsize=16)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('合同项数', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)  # 半透明网格线

    # 坐标轴格式设置
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # 年度格式
    plt.xticks(rotation=45)  # 日期旋转

    # 输出结果
    plt.savefig(f'out_file/figs/{country}_预测结果.png', dpi=300)  # 保存高清图
    plt.show()


def save_results_to_excel(country, future_predictions, years_to_predicted, series):
    """保存预测结果到Excel文件（自动生成未来日期序列）
    Args:
        country: 国家名称（用于文件名）
        future_predictions: 未来预测值数组
        years_to_predicted: 预测步长（年数）
        series: 原始时间序列（用于获取最后日期）
    """
    # 生成未来日期序列（从原始数据最后日期开始）
    future_dates = pd.date_range(
        start=series.index[-1] + pd.DateOffset(years=1),  # 从最后日期+1年开始
        periods=years_to_predicted,
        freq='YS'  # 年初为日期节点
    )

    # 创建并保存DataFrame
    pd.DataFrame({
        '日期': future_dates.strftime('%Y-%m-%d'),  # 统一日期格式
        '预测值': np.round(future_predictions).astype(int)  # 取整处理
    }).to_excel(
        f'out_file/excels/{country}_未来预测.xlsx',  # 按国家命名文件
        index=False  # 不保存行索引
    )


def main():
    """主执行函数：完成从数据加载到预测结果输出的全流程"""
    # 1. 数据准备阶段
    country = input("请输入需要预测的国家/地区：")
    df = read_excel('深圳签订外贸合同项数数据1990~2023.xlsx', index_col=0)
    series = df[country].dropna()  # 获取指定国家数据并去除空值

    # 2. 差分阶数确定（特殊国家使用预设值）
    if country == "其他" or "美国":
        d = 1
    elif country == "泰国":
        d = 0
    else:
        d = find_optimal_diff(series)  # 自动寻找最优差分阶数
    diff_series = difference(series, d)  # 执行差分

    # 3. 数据预处理
    look_back = 5  # 增大时间窗口以捕捉更长周期模式
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(diff_series.values.reshape(-1, 1))  # 归一化

    # 4. 数据集构建（转换为监督学习格式）
    X, Y = create_dataset(scaled_data.flatten(), look_back)
    X = X.reshape(X.shape[0], 1, X.shape[1])  # 调整LSTM输入形状 [样本数, 时间步, 特征数]

    # 5. 训练测试集划分（90%训练）
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # 6. 模型构建与训练
    model = build_lstm_model(look_back, neurons=200)
    early_stop = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

    model.fit(X_train, Y_train,
              epochs=1000,
              batch_size=1,  # 在线学习模式
              validation_data=(X_test, Y_test),
              callbacks=[early_stop],
              verbose=1,
              shuffle=False)  # 时间序列禁止打乱

    # 7. 测试集预测与逆转换
    test_predict = model.predict(X_test, verbose=0)
    test_predict = scaler.inverse_transform(test_predict)  # 反归一化

    # 逆差分还原预测值
    test_predict_undiff = []
    history_values = series.values.tolist()  # 保留历史真实值
    for i in range(len(test_predict)):
        pred = inverse_difference(history_values, test_predict[i][0], d)
        test_predict_undiff.append(pred)
        history_values.append(pred)  # 模拟实时预测场景

    # 8. 未来预测（添加随机噪声增强鲁棒性）
    years_to_predicted = 20 # 预测未来年数
    future_predictions = []
    current_batch = scaled_data[-look_back:].reshape(1, 1, look_back)  # 初始化预测批次

    for _ in range(years_to_predicted):
        current_pred = model.predict(current_batch, verbose=0)[0]
        raw_pred = scaler.inverse_transform(current_pred.reshape(-1, 1))[0][0]  # 反归一化
        undiff_pred = inverse_difference(history_values, raw_pred, d)
        noise = np.random.normal(0, abs(undiff_pred) * 0.05)  # 添加5%高斯噪声
        future_predictions.append(max(undiff_pred + noise, 0))  # 确保非负

        # 更新预测窗口（滑动窗口机制）
        current_batch = np.append(current_batch[0][0][1:], current_pred).reshape(1, 1, look_back)
        history_values.append(undiff_pred)

    # 9. 评估指标输出
    actual = series[-len(test_predict_undiff):].values
    print(f"\n{country} - {d}阶差分")
    print(f'RMSE: {sqrt(mean_squared_error(actual, test_predict_undiff)):.2f}')
    print(f'MAE: {mean_absolute_error(actual, test_predict_undiff):.2f}')
    print(f'MAPE: {np.mean(np.abs((actual - test_predict_undiff) / actual)) * 100:.2f}%')

    # 10. 结果可视化与保存
    plot_results(series.values, test_predict_undiff, future_predictions, country)
    save_results_to_excel(country, future_predictions, years_to_predicted, series)

if __name__ == '__main__':
    main()
    """
    '中国香港、澳门', '中国台湾', '新加坡', '韩国', '日本', 
    '泰国', '澳大利亚', '马来西亚', '美国', '加拿大', 
    '德国', '法国', '英国', '瑞士', '荷兰', '其他'
    """