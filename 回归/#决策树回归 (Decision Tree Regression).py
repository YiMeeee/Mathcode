#决策树回归 (Decision Tree Regression)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

def decision_tree_regression(X, y, max_depth=5):
    regressor = DecisionTreeRegressor(max_depth=max_depth)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    plt.figure(figsize=(10, 8))
    plot_tree(regressor, filled=True)
    plt.title("Decision Tree Regression")
    plt.show()
    
    # Plot regression curve
    X_grid = np.arange(min(X), max(X), 0.01)[:, np.newaxis]
    y_grid_pred = regressor.predict(X_grid)
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X_grid, y_grid_pred, color='red', label='Decision Tree regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Decision Tree Regression')
    plt.legend()
    plt.show()

def main():
    file_path = input("Please enter the path to your data file (Excel or CSV): ")
    sheet_name = input("Please enter the sheet name (if applicable, otherwise press Enter): ") or None
    try:
        X, y = read_data(file_path, sheet_name=sheet_name)
        X, y = preprocess_data(X, y)
        
        max_depth = int(input("Please enter the max_depth value (default 5): ") or 5)
        
        decision_tree_regression(X, y, max_depth=max_depth)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

#升级版
"""
# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# 创建一个虚拟数据集
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16))  # 添加噪声

# 训练回归模型
regressor = DecisionTreeRegressor(max_depth=5)
regressor.fit(X, y)

# 绘制决策树图像
plt.figure(figsize=(10, 8))
plot_tree(regressor, filled=True)
plt.title("Decision Tree Regression")
plt.show()
"""