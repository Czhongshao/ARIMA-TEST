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
    """自动寻找最优差分阶数"""
    best_d = 0
    current_series = series.copy()
    
    for d in range(0, max_diff + 1):
        test_series = current_series if d == 0 else difference(current_series, 1)
        p_value = adfuller(test_series.dropna())[1]
        if p_value < 0.05:
            best_d = d
            break
        current_series = test_series
    
    return best_d

def difference(series, d=1):
    """递归差分函数"""
    for _ in range(d):
        series = series.diff().dropna()
    return series

def inverse_difference(history, yhat_diff, d):
    """多阶逆差分（确保非负）"""
    yhat = yhat_diff
    for i in range(d):
        yhat += history[-(i + 1)]
    return max(yhat, 0)

def create_dataset(data, look_back=3):
    """创建时间序列数据集"""
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

def build_lstm_model(look_back, neurons=256):
    """改进的LSTM模型架构"""
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=(1, look_back)),
        Dropout(0.3),
        LSTM(neurons, return_sequences=True),
        Dropout(0.3),
        LSTM(neurons),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

def plot_results(true, pred, future_pred, country):
    """增强的可视化函数"""
    plt.figure(figsize=(14, 7))
    dates = pd.date_range(start='1990', periods=len(true), freq='YE')
    plt.plot(dates, true, label='实际值', color='blue', marker='o')
    
    test_dates = dates[-len(pred):]
    plt.plot(test_dates, pred, label='测试预测', color='red', linestyle='--', marker='x')
    
    future_dates = pd.date_range(start=dates[-1] + pd.DateOffset(years=1), periods=len(future_pred), freq='YE')
    plt.plot(future_dates, future_pred, label='未来预测', color='green', linestyle='--', marker='^')
    
    plt.title(f'{country} - 外贸合同签订项数预测', fontsize=16)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('合同项数', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.savefig(f'out_file/figs/{country}_预测结果.png', dpi=300)
    plt.show()

def save_results_to_excel(country, future_predictions, future_steps, series):
    """保存预测结果到Excel文件"""
    future_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(years=1),
                                periods=future_steps, freq='YS')
    pd.DataFrame({
        '日期': future_dates.strftime('%Y-%m-%d'),
        '预测值': np.round(future_predictions).astype(int)
    }).to_excel(f'out_file/excels/{country}_未来预测.xlsx', index=False)

def main():
    country = input("请输入需要预测的国家/地区：")
    df = read_excel('深圳签订外贸合同项数数据1990~2023.xlsx', index_col=0)
    series = df[country].dropna()
    if country == "其他" or "美国": d = 1
    elif country == "泰国": d = 0 
    else: find_optimal_diff(series)
    diff_series = difference(series, d)
    
    look_back = 5  # 增加历史窗口
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(diff_series.values.reshape(-1, 1))
    
    X, Y = create_dataset(scaled_data.flatten(), look_back)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    train_size = int(len(X) * 0.9)  # 增加训练数据比例
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    model = build_lstm_model(look_back, neurons=200)
    early_stop = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
    
    model.fit(X_train, Y_train, epochs=1000, batch_size=1, validation_data=(X_test, Y_test), callbacks=[early_stop], verbose=1, shuffle=False)
    
    test_predict = model.predict(X_test, verbose=0)
    test_predict = scaler.inverse_transform(test_predict)
    
    test_predict_undiff = []
    history_values = series.values.tolist()
    for i in range(len(test_predict)):
        pred = inverse_difference(history_values, test_predict[i][0], d)
        test_predict_undiff.append(pred)
        history_values.append(pred)
    
    future_steps = 20
    future_predictions = []
    current_batch = scaled_data[-look_back:].reshape(1, 1, look_back)
    for _ in range(future_steps):
        current_pred = model.predict(current_batch, verbose=0)[0]
        raw_pred = scaler.inverse_transform(current_pred.reshape(-1, 1))[0][0]
        undiff_pred = inverse_difference(history_values, raw_pred, d)
        noise = np.random.normal(0, abs(undiff_pred) * 0.05)
        future_predictions.append(max(undiff_pred + noise, 0))
        current_batch = np.append(current_batch[0][0][1:], current_pred).reshape(1, 1, look_back)
        history_values.append(undiff_pred)
    
    actual = series[-len(test_predict_undiff):].values
    print(f"{country} - {d}阶差分")
    print(f'RMSE: {sqrt(mean_squared_error(actual, test_predict_undiff)):.2f}')
    print(f'MAE: {mean_absolute_error(actual, test_predict_undiff):.2f}')
    print(f'MAPE: {np.mean(np.abs((actual - test_predict_undiff)/actual)) * 100:.2f}%')
    
    plot_results(series.values, test_predict_undiff, future_predictions, country)
    save_results_to_excel(country, future_predictions, future_steps, series)

if __name__ == '__main__':
    main()
    """
    '中国香港、澳门', '中国台湾', '新加坡', '韩国', '日本', 
    '泰国', '澳大利亚', '马来西亚', '美国', '加拿大', 
    '德国', '法国', '英国', '瑞士', '荷兰', '其他'
    """