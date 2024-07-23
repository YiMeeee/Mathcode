import numpy as np
import pandas as pd#需要安装的库
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 读取Excel文件中的数据
file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的Excel表名
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel文件中有两列数据：X 和 y
X = data['X'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)

# 线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)

# 绘图
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

"""
#下列代码包含了去除了缺失项的功能能，自动检测Excel文件中的列名，在绘图中显示回归方程和R²值。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def read_excel_data(file_path, sheet_name='Sheet1'):
    # 读取Excel文件
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def preprocess_data(data, x_column, y_column):
    # 处理缺失数据（删除包含NaN的行）
    data = data[[x_column, y_column]].dropna()
    X = data[x_column].values.reshape(-1, 1)
    y = data[y_column].values.reshape(-1, 1)
    return X, y

def perform_linear_regression(X, y):
    # 线性回归模型
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred = lin_reg.predict(X)
    return lin_reg, y_pred

def plot_results(X, y, y_pred, lin_reg, x_column, y_column):
    # 绘图
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Regression line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('Linear Regression')
    
    # 添加回归方程和R²值
    slope = lin_reg.coef_[0][0]
    intercept = lin_reg.intercept_[0]
    r2 = r2_score(y, y_pred)
    equation_text = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.2f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top')
    
    plt.legend()
    plt.show()

def main(file_path, sheet_name='Sheet1', x_column='X', y_column='y'):
    data = read_excel_data(file_path, sheet_name)
    X, y = preprocess_data(data, x_column, y_column)
    lin_reg, y_pred = perform_linear_regression(X, y)
    plot_results(X, y, y_pred, lin_reg, x_column, y_column)

if __name__ == "__main__":
    file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径
    sheet_name = 'Sheet1'  # 替换为你的Excel表名
    x_column = 'X'  # 替换为X轴列名
    y_column = 'y'  # 替换为Y轴列名
    main(file_path, sheet_name, x_column, y_column)
"""