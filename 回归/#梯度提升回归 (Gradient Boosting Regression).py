#梯度提升回归 (Gradient Boosting Regression)
#升级版本
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
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

def gradient_boosting_regression(X, y, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0):
    est = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                    max_depth=max_depth, random_state=random_state, loss='ls')
    est.fit(X, y)
    y_pred = est.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Generate test data
    X_test = np.linspace(min(X), max(X), 500).reshape(-1, 1)
    y_test_pred = est.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', s=30, marker="o", alpha=0.5, label='Observations')
    plt.plot(X_test, true_function(X_test), color='blue', label='True function')
    plt.plot(X_test, y_test_pred, '--', color='navy', label='Predicted function')
    plt.title('Gradient Boosting Regression')
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def true_function(x):
    return np.sin(x) + np.sin(10 * x)

def main():
    file_path = input("Please enter the path to your data file (Excel or CSV): ")
    sheet_name = input("Please enter the sheet name (if applicable, otherwise press Enter): ") or None
    try:
        X, y = read_data(file_path, sheet_name=sheet_name)
        X, y = preprocess_data(X, y)
        
        n_estimators = int(input("Please enter the number of estimators (default 100): ") or 100)
        learning_rate = float(input("Please enter the learning rate (default 0.1): ") or 0.1)
        max_depth = int(input("Please enter the max depth (default 3): ") or 3)
        random_state = int(input("Please enter the random state (default 0): ") or 0)
        
        gradient_boosting_regression(X, y, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# 创建一个非线性函数作为模拟数据
def true_function(x):
    return np.sin(x) + np.sin(10 * x)

# 生成模拟数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = true_function(X).ravel()
dy = 0.5 + 1.0 * np.random.rand(200)
y += np.random.normal(0, dy)

# 拟合梯度提升回归模型
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=3, random_state=0, loss='ls')
est.fit(X, y)

# 生成预测数据
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = est.predict(X_test)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.errorbar(X.ravel(), y, dy, fmt='o', alpha=0.5, label='Observations')
plt.plot(X_test, true_function(X_test), label='True function', color='blue')
plt.plot(X_test, y_pred, '--', color='navy', label='Predicted function')
plt.fill_between(X_test.ravel(), y_pred - est.estimators_[0][0].predict(X_test), y_pred + est.estimators_[0][0].predict(X_test), color='red', alpha=0.6, label='Uncertainty')
plt.title('Gradient Boosting Regression')
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
"""