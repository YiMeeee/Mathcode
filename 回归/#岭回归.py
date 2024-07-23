#岭回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

def read_data(file_path=None, x_column='X', y_column='y', sheet_name='Sheet1'):
    if file_path:
        # 读取Excel文件中的数据
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        # 提取X和y列数据
        X = data[x_column].values.reshape(-1, 1)
        y = data[y_column].values.reshape(-1, 1)
    else:
        # 生成随机数据,这里意味着上述模型如果有错误，将会采用下列的模型，防止有bug导致整个程序无法运作了
        np.random.seed(0)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

def perform_ridge_regression(X, y, alpha=1):
    # 岭回归模型
    ridge_reg = Ridge(alpha=alpha, solver="cholesky")
    ridge_reg.fit(X, y)
    y_pred = ridge_reg.predict(X)
    return ridge_reg, y_pred

def plot_results(X, y, y_pred, x_column='X', y_column='y'):
    # 绘图
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Ridge regression line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('Ridge Regression')
    plt.legend()
    plt.show()

def main(file_path=None, x_column='X', y_column='y', sheet_name='Sheet1', alpha=1):
    X, y = read_data(file_path, x_column, y_column, sheet_name)
    ridge_reg, y_pred = perform_ridge_regression(X, y, alpha)
    plot_results(X, y, y_pred, x_column, y_column)

if __name__ == "__main__":
    file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径或设为None使用随机数据
    x_column = 'X'  # 替换为X轴列名
    y_column = 'y'  # 替换为Y轴列名
    sheet_name = 'Sheet1'  # 替换为你的Excel表名
    alpha = 1  # 岭回归的alpha参数
    main(file_path, x_column, y_column, sheet_name, alpha)

"""
下列代码包含了去除了缺失项的功能能，自动检测Excel文件中的列名，在绘图中显示回归方程和R²值。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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

def perform_ridge_regression(X, y, alpha=1):
    # 岭回归模型
    ridge_reg = Ridge(alpha=alpha, solver="cholesky")
    ridge_reg.fit(X, y)
    y_pred = ridge_reg.predict(X)
    return ridge_reg, y_pred

def plot_results(X, y, y_pred, ridge_reg, x_column, y_column):
    # 绘图
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Ridge regression line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('Ridge Regression')
    
    # 添加回归方程和R²值
    slope = ridge_reg.coef_[0][0]
    intercept = ridge_reg.intercept_[0]
    r2 = r2_score(y, y_pred)
    equation_text = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.2f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top')
    
    plt.legend()
    plt.show()

def main(file_path=None, sheet_name='Sheet1', x_column='X', y_column='y', alpha=1):
    if file_path:
        data = read_excel_data(file_path, sheet_name)
        X, y = preprocess_data(data, x_column, y_column)
    else:
        # 生成随机数据
        np.random.seed(0)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)

    ridge_reg, y_pred = perform_ridge_regression(X, y, alpha)
    plot_results(X, y, y_pred, ridge_reg, x_column, y_column)

if __name__ == "__main__":
    file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径或设为None使用随机数据
    sheet_name = 'Sheet1'  # 替换为你的Excel表名
    x_column = 'X'  # 替换为X轴列名
    y_column = 'y'  # 替换为Y轴列名
    alpha = 1  # 岭回归的alpha参数
    main(file_path, sheet_name, x_column, y_column, alpha)
"""