#弹性网络回归 (Elastic Net Regression)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

#从excel获取文件
# 读取Excel文件中的数据
file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的Excel表名
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel文件中有两列数据：X 和 y
X = data['X'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)


# 弹性网络回归模型
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
y_pred = elastic_net.predict(X)

# 绘图
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Elastic Net regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Elastic Net Regression')
plt.legend()
plt.show()

"""
#升级版本：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import os

def read_data(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == '.xlsx' or ext == '.xls':
        data = pd.read_excel(file_path)
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

def elastic_net_regression(X, y, alpha=0.1, l1_ratio=0.5):
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error: {mse}')

    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Elastic Net regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Elastic Net Regression')
    plt.legend()
    plt.show()

def main():
    file_path = input("Please enter the path to your data file (Excel or CSV): ")
    try:
        X, y = read_data(file_path)
        X, y = preprocess_data(X, y)
        
        alpha = float(input("Please enter the alpha value (default 0.1): ") or 0.1)
        l1_ratio = float(input("Please enter the l1_ratio value (default 0.5): ") or 0.5)
        
        elastic_net_regression(X, y, alpha=alpha, l1_ratio=l1_ratio)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


"""