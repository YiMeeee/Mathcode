#多项式回归 (Polynomial Regression)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#从excel获取文件
# 读取Excel文件中的数据
file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的Excel表名
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel文件中有两列数据：X 和 y
X = data['X'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)


# 多项式特征变换
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
y_pred = lin_reg.predict(X_poly)

# 绘图
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Polynomial regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
"""
#升级版本

"""