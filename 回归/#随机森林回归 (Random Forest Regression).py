#随机森林回归 (Random Forest Regression)

#升级版本
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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

def random_forest_regression(X, y, n_estimators=100, random_state=42):
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Plotting the Random Forest regression results
    X_test = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    y_test_pred = regressor.predict(X_test)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X, y, color="blue", s=30, marker="o", label="Training data")
    plt.plot(X_test, y_test_pred, color="red", label="Predictions")
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Random Forest Regression')
    plt.legend()
    plt.show()
    
    # Visualize one of the decision trees in the random forest
    estimator = regressor.estimators_[0]
    plt.figure(figsize=(15, 10))
    from sklearn.tree import plot_tree
    plot_tree(estimator, filled=True, feature_names=['Feature'])
    plt.title('Example Decision Tree from Random Forest')
    plt.show()

def main():
    file_path = input("Please enter the path to your data file (Excel or CSV): ")
    sheet_name = input("Please enter the sheet name (if applicable, otherwise press Enter): ") or None
    try:
        X, y = read_data(file_path, sheet_name=sheet_name)
        X, y = preprocess_data(X, y)
        
        n_estimators = int(input("Please enter the number of trees (n_estimators, default 100): ") or 100)
        random_state = int(input("Please enter the random state (default 42): ") or 42)
        
        random_forest_regression(X, y, n_estimators=n_estimators, random_state=random_state)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 训练随机森林回归模型
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X, y)

# 预测结果
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = regressor.predict(X_test)

# 绘制随机森林中的一棵决策树图像
estimator = regressor.estimators_[0]

plt.figure(figsize=(10, 8))
plt.scatter(X, y, color="b", s=30, marker="o", label="training data")
plt.plot(X_test, y_pred, color="r", label="predictions")
plt.title("Random Forest Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend(loc="upper left")

# 可视化一棵决策树
plt.figure(figsize=(15, 10))
from sklearn.tree import plot_tree
plot_tree(estimator, filled=True, feature_names=['Feature'])
plt.title("Example Decision Tree from Random Forest")
plt.show()
"""