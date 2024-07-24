#支持向量回归 (Support Vector Regression, SVR)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR

#从excel获取文件
# 读取Excel文件中的数据
file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的Excel表名
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel文件中有两列数据：X 和 y
X = data['X'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)


# 支持向量回归模型
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X, y.ravel())
y_pred = svr.predict(X)

# 绘图
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='SVR regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

"""
#升级版本
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
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

def support_vector_regression(X, y, kernel='rbf', C=100, epsilon=0.1):
    svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr.fit(X, y.ravel())
    y_pred = svr.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error: {mse}')

    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='SVR regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

def main():
    file_path = input("Please enter the path to your data file (Excel or CSV): ")
    sheet_name = input("Please enter the sheet name (if applicable, otherwise press Enter): ") or None
    try:
        X, y = read_data(file_path, sheet_name=sheet_name)
        X, y = preprocess_data(X, y)
        
        kernel = input("Please enter the kernel type (default 'rbf'): ") or 'rbf'
        C = float(input("Please enter the C value (default 100): ") or 100)
        epsilon = float(input("Please enter the epsilon value (default 0.1): ") or 0.1)
        
        support_vector_regression(X, y, kernel=kernel, C=C, epsilon=epsilon)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


    

#升级版
# 生成随机数据
# np.random.seed(0)
# X = 6 * np.random.rand(100, 1) - 3
# y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
"""