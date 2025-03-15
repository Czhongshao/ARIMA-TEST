# 导入必要的库
from pandas import read_csv
from datetime import datetime, timedelta
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 定义日期解析函数，用于将字符串日期转换为 datetime 对象
def parser(x):
    return datetime.strptime(x, '%Y')  # 直接将年份字符串解析为日期

# 读取数据
# 从 '111.csv' 文件中读取数据，指定第一列为日期索引，并解析日期
series = read_csv('原数据转成csv模式.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)

# 选择“中国香港、澳门”的数据列进行分析
"""

这里的数据列名称也要改。不同的国家/地区要用不同的数据集

"""
series = series['中国香港、澳门']

# 获取数据的值部分（忽略索引），并将其存储为 NumPy 数组
X = series.values

# 划分训练集和测试集
# 使用 60% 的数据作为训练集，剩余部分作为测试集
size = int(len(X) * 0.60)
train, test = X[0:size], X[size:len(X)]

# 初始化历史数据列表，用于存储训练集中的数据
history = [x for x in train]

# 初始化预测结果列表
predictions = list()

# 对测试集中的每个时间点进行预测
for t in range(len(test)):
    # 使用历史数据拟合 ARIMA 模型，指定模型参数 order=(5,1,0)
    """
    
    在下面这一行更改参数
    
    """
    model = ARIMA(history, order=(0, 2, 2))     #它是一个三元组 (p【PACF】, d, q【ACF】)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print(f'predicted={yhat}, expected={obs}')

# 计算预测结果的均方误差（MSE）
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# 使用整个数据集拟合模型，预测未来5年的数据
model = ARIMA(X, order=(5, 1, 0))
model_fit = model.fit()
future_forecast = model_fit.forecast(steps=5)  # 预测未来5年的数据

# 创建未来年份的索引
last_year = series.index[-1].year
future_years = [datetime(year=last_year + i + 1, month=1, day=1) for i in range(5)]

# 绘制实际值、测试集预测值和未来预测值的对比图
pyplot.figure(figsize=(10, 6))  # 设置图像大小
pyplot.plot(series.index, X, label='Actual', marker='o')  # 绘制实际值曲线
pyplot.plot(series.index[size:], predictions, color='red', label='Predicted (Test)', linestyle='--', marker='x')  # 绘制测试集预测值曲线
pyplot.plot(future_years, future_forecast, color='green', label='Future Forecast', linestyle='-.', marker='^')  # 绘制未来预测值曲线
pyplot.legend()  # 添加图例
pyplot.title('ARIMA Model Prediction for Singapore')  # 添加标题
pyplot.xlabel('Year')  # 添加X轴标签
pyplot.ylabel('Number of Contracts')  # 添加Y轴标签
pyplot.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

# 在每个数据点上显示数值
for i, value in enumerate(X):
    pyplot.text(series.index[i], value, f'{value:.0f}', ha='center', va='bottom', fontsize=8, color='blue')

for i, value in enumerate(predictions):
    pyplot.text(series.index[size:][i], value, f'{value:.0f}', ha='center', va='top', fontsize=8, color='red')

for i, value in enumerate(future_forecast):
    pyplot.text(future_years[i], value, f'{value:.0f}', ha='center', va='bottom', fontsize=8, color='green')

pyplot.show()  # 显示图像