import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, concat, read_excel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from math import sqrt
import matplotlib as mpl
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
from matplotlib.dates import DateFormatter

# 计算最优的差分阶数
def find_best_difference(series):
    p_value_threshold = 0.05  # 设定平稳性的 p 值阈值
    diff_series = series.copy()
    d = 0
    while adfuller(diff_series)[1] > p_value_threshold:
        d += 1
        diff_series = difference(series, d)
    return d

# 差分处理
def difference(dataset, interval=1):
    return Series([dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))])

# 逆差分处理
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# 转换时间序列数据为监督学习格式
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# 归一化数据
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train)
    return scaler, scaler.transform(train), scaler.transform(test)

# 逆归一化
def invert_scale(scaler, X, y):
    array = np.array([*X, y]).reshape(1, -1)
    return scaler.inverse_transform(array)[0, -1]

# 训练 LSTM 模型
def fit_lstm(train, batch_size, epochs, neurons):
    X, y = train[:, :-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(neurons, return_sequences=True),
        Dropout(0.2),
        LSTM(neurons, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    for _ in range(epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
    
    return model

def make_predictions_v2(model, scaler, data, raw_values, steps_ahead):
    predictions = []
    last_window = data[-1, :-1].reshape(1, 1, -1)  # 确保取的是 2023 年的数据
    
    last_actual_value = raw_values[-1]  # 2023 年真实值

    for i in range(steps_ahead):
        yhat = model.predict(last_window, batch_size=1)[0, 0]
        yhat = invert_scale(scaler, last_window.flatten(), yhat)
        yhat = inverse_difference(raw_values, yhat, 1)  # 使用2023年数据做逆变换

        # 添加一定的波动性
        noise = np.random.normal(0, 0.02 * yhat)
        yhat = max(yhat + noise, 0)
        predictions.append(yhat)

        # 使用新的 yhat 作为输入，确保不跳过 2024
        new_input = np.append(last_window.flatten()[1:], yhat - last_actual_value)
        last_window = new_input.reshape(1, 1, -1)
        
        last_actual_value = yhat  # 更新上一年的真实值

    return predictions


# 读取数据
series = read_excel('深圳签订外贸合同项数数据1990~2023.xlsx', header=0, index_col=0)
need_to_predict = "英国"
raw_values = series[need_to_predict].values

# 进行对数变换，减少数据波动
raw_values_log = np.log1p(raw_values)

# 计算最优差分阶数并进行差分转换
best_d = find_best_difference(raw_values_log)
diff_values = difference(raw_values_log, best_d)

# 转换为监督学习数据
supervised_values = timeseries_to_supervised(diff_values, 1).values

# 分割训练集和测试集
train, test = supervised_values[:-12], supervised_values[-12:]

# 数据缩放
scaler, train_scaled, test_scaled = scale(train, test)

# 训练 LSTM 模型
lstm_model = fit_lstm(train_scaled, batch_size=16, epochs=500, neurons=50)

# 计算测试集预测
test_predictions = make_predictions_v2(lstm_model, scaler, test_scaled, raw_values, len(test_scaled))
rmse = sqrt(mean_squared_error(raw_values[-12:], test_predictions))
print(f'Test RMSE: {rmse:.3f}')

# 预测未来 20 年
future_steps = 20
future_predictions = make_predictions_v2(lstm_model, scaler, test_scaled, raw_values, future_steps)


# 绘图函数
def plot_results(series, test, predictions, future_dates=None, future_predictions=None, title=""):
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series[need_to_predict].values, color='blue', label='Original Data')
    
    if predictions is not None:
        plt.plot(series.index[-len(test):], predictions, color='red', linestyle='--', label='Test Predictions')
    
    if future_dates is not None and future_predictions is not None:
        plt.plot(future_dates, future_predictions, color='green', linestyle='--', label='Future Predictions')
    
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('项数')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.tight_layout()
    plt.show()


# 计算评估指标
mse = mean_squared_error(raw_values[-12:], test_predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(raw_values[-12:], test_predictions)
mape = np.mean(np.abs((raw_values[-12:] - test_predictions) / raw_values[-12:])) * 100

print(f"Test MSE: {mse:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE: {mae:.3f}")
print(f"Test MAPE: {mape:.3f}%")

# 绘制测试预测结果
plot_results(series, test, test_predictions, title=need_to_predict)

# 生成未来时间序列
future_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(years=1), periods=future_steps, freq='YS')
plot_results(series, test, None, future_dates, future_predictions, title=f"{need_to_predict} - Future Prediction")