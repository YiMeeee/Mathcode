#贝叶斯回归 (Bayesian Regression)
#升级版本
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
import os

def read_data(file_path, sheet_name=None):
    ext = os.path.splitext(file_path)[1]
    if ext == '.xlsx' or ext == '.xls':
        data = pd.read_excel(file_path, sheet_name=sheet_name)
    elif ext == '.csv':
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Please use an Excel or CSV file.")
    
    if 'X' in data.columns and 'y' in data.columns:
        X = data[['X']].values
        y = data[['y']].values
        return X, y
    else:
        raise ValueError("The file should contain columns named 'X' and 'y'.")

def preprocess_data(X, y):
    # Check for missing values
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print("Missing values found. Imputing with mean values.")
        X = np.nan_to_num(X, nan=np.nanmean(X))
        y = np.nan_to_num(y, nan=np.nanmean(y))
    return X, y

def bayesian_regression(X, y, degree=5):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    clf = BayesianRidge(compute_score=True)
    clf.fit(X_poly, y)
    
    y_pred = clf.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Generate test data
    X_test = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_test_poly = poly.transform(X_test)
    y_test_pred, y_std = clf.predict(X_test_poly, return_std=True)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(X.ravel(), y, yerr=0.1, fmt='o', alpha=0.5, label='Observations')
    plt.plot(X_test, true_function(X_test), color='green', label='True function')
    plt.plot(X_test, y_test_pred, '--', color='navy', label='Predicted function')
    plt.fill_between(X_test.ravel(), y_test_pred - y_std, y_test_pred + y_std, color='red', alpha=0.6, label='Uncertainty')
    plt.title('Bayesian Regression with Polynomial Features')
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def true_function(x):
    return np.sin(x) + np.sin(2 * x) + np.sin(3 * x)

def main():
    file_path = input("Please enter the path to your data file (Excel or CSV): ")
    sheet_name = input("Please enter the sheet name (if applicable, otherwise press Enter): ") or None
    try:
        X, y = read_data(file_path, sheet_name=sheet_name)
        X, y = preprocess_data(X, y)
        
        degree = int(input("Please enter the degree of the polynomial features (default 5): ") or 5)
        
        bayesian_regression(X, y, degree=degree)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

"""
#简易版，可以供给参考进行修改
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge

# 创建一个非线性函数作为模拟数据
def true_function(x):
    return np.sin(x) + np.sin(2 * x) + np.sin(3 * x)

# 生成模拟数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = true_function(X).ravel()
dy = 0.5 + 1.0 * np.random.rand(200)
y += np.random.normal(0, dy)

# 贝叶斯线性回归模型
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
clf = BayesianRidge(compute_score=True)
clf.fit(X_poly, y)

# 生成预测数据
X_test = np.linspace(0, 5, 100)[:, np.newaxis]
y_pred, y_std = clf.predict(poly.transform(X_test), return_std=True)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.errorbar(X.ravel(), y, dy, fmt='o', alpha=0.5, label='Observations')
plt.plot(X_test, true_function(X_test), color='green', label='True function')
plt.plot(X_test, y_pred, '--', color='navy', label='Predicted function')
plt.fill_between(X_test.ravel(), y_pred - y_std, y_pred + y_std, color='red', alpha=0.6, label='Uncertainty')
plt.title('Bayesian Regression with Polynomial Features')
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
"""