%% 数据导入
clc; clear; close all;
data_1 = readtable("深圳签订外贸合同项数数据1990~2023.xlsx", "VariableNamingRule", "preserve");
disp(data_1);

% 获取所有数值列的列名
numeric_vars = varfun(@isnumeric, data_1, 'OutputFormat', 'uniform');
numeric_col_names = data_1.Properties.VariableNames(numeric_vars);

% 提取数值列的数据
numeric_data = table2array(data_1(:, numeric_vars));

% 绘制箱线图
figure; % 创建一个新的图形窗口
boxplot(numeric_data);

% 设置 X 轴标签为列名
set(gca, 'XTickLabel', numeric_col_names);

% 添加标题和标签
title('原始数据的箱线图');
xlabel('数据列');
ylabel('数值');
%% 对每个数值列进行三倍标准差处理
for i = 1:length(numeric_col_names)
    col_name = numeric_col_names{i};
    
    % 计算均值和标准差
    mean_data = mean(data_1.(col_name), 'omitnan');
    std_data = std(data_1.(col_name), 0, 'omitnan');
    
    % 确定异常值的范围
    lower_bound = mean_data - 3 * std_data;
    upper_bound = mean_data + 3 * std_data;
    
    % 将异常值替换为 NaN
    data_1.(col_name)(data_1.(col_name) < lower_bound | data_1.(col_name) > upper_bound) = NaN;
end

%% 对 NaN 值进行整数均值填充
for i = 1:length(numeric_col_names)
    col_name = numeric_col_names{i};
    % 计算列的均值并四舍五入到整数
    mean_value = round(mean(data_1.(col_name), 'omitnan'));
    % 使用整数均值填充 NaN 值
    data_1.(col_name) = fillmissing(data_1.(col_name), 'constant', mean_value);
end

%% 显示处理后的数据
disp(data_1);