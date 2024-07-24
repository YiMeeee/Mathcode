import numpy as np
import pandas as pd#需要安装的库
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

#从excel获取文件
# 读取Excel文件中的数据
file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的Excel表名
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel文件中有两列数据：X 和 y
X = data['X'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)


# 拉索回归模型
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
y_pred = lasso_reg.predict(X)

# 绘图
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Lasso regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Lasso Regression')
plt.legend()
plt.show()
"""
老样子这里放置着升级版
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def read_data(file_path=None, x_column='X', y_column='y', sheet_name='Sheet1'):
    if file_path:
        # 读取Excel文件中的数据
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        # 提取X和y列数据
        X = data[x_column].values.reshape(-1, 1)
        y = data[y_column].values.reshape(-1, 1)
    else:
        # 生成随机数据
        np.random.seed(0)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

def perform_lasso_regression(X, y, alpha=0.1):
    # 拉索回归模型
    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X, y)
    y_pred = lasso_reg.predict(X)
    return lasso_reg, y_pred

def plot_results(X, y, y_pred, x_column='X', y_column='y'):
    # 绘图
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Lasso regression line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('Lasso Regression')
    plt.legend()
    plt.show()

def main(file_path=None, x_column='X', y_column='y', sheet_name='Sheet1', alpha=0.1):
    X, y = read_data(file_path, x_column, y_column, sheet_name)
    lasso_reg, y_pred = perform_lasso_regression(X, y, alpha)
    plot_results(X, y, y_pred, x_column, y_column)

if __name__ == "__main__":
    file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径或设为None使用随机数据
    x_column = 'X'  # 替换为X轴列名
    y_column = 'y'  # 替换为Y轴列名
    sheet_name = 'Sheet1'  # 替换为你的Excel表名
    alpha = 0.1  # Lasso回归的alpha参数
    main(file_path, x_column, y_column, sheet_name, alpha)

"""