# 1. 创建时间序列Y
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import numpy as np
Y = np.array([ 0, 9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 98, 104, 105, 116, 149, 156, 247, 249, 250, 337  ])
N = len(Y)
print(N)
print(Y)
plt.plot(Y, 'o-', label='初始值')
plt.legend()

# 用于创建轨迹矩阵，公式（1）
def trajectory_matrix(X, k):
    N = len(X)
    L = N - k + 1
    out = []
    for i in range(L):
        out.append(X[i:k+i])
    out = np.asarray(out)
    return out

# 创建轨迹矩阵
# N=27, K=23, L=N-K+1
K = 23
L = N - K + 1
print(f"L={L}, K={K}")
X = trajectory_matrix(Y, K)
print(X.shape)

# 奇异值分解，以下numpy实现的svd已经对奇异值降序排序了，对应奇异向量也已经重新排序了。
U, s, VT = np.linalg.svd(X)

# rank of X
r = np.linalg.matrix_rank(X)
print(f"rank={r}")

# 分组，这里默认每个奇异值作为一个组，所以从0遍历到r-1，共r个分组。
Xs = []
for i in range(r):
    t = s[i] * U[:, i][:,None] @ VT[i,:][:,None].T
    Xs.append(t)
print(len(Xs))
X1 = Xs[0]
X2 = Xs[1]
X3 = Xs[2]
X4 = Xs[3]

# 用于计算矩阵反对角线的均值
def average_anti_diag(X):
    X = np.asarray(X)
    out = [np.mean(X[::-1, :].diagonal(i)) for i in range(-X.shape[0]+1, X.shape[1])]
    return np.asarray(out)

# 计算各个分组的子矩阵的反对角线的均值，并得到一个子序列
Y1 = average_anti_diag(X1)
Y2 = average_anti_diag(X2)
Y3 = average_anti_diag(X3)
Y4 = average_anti_diag(X4)

# 将得到的子序列求和得到重构的时间序列
dta = Y1 + Y2 + Y3 + Y4

model = pm.auto_arima(dta, start_p=1, start_q=1,
                           max_p=8, max_q=8, m=1,
                           start_P=0, seasonal=False,
                           max_d=3, trace=True,
                           information_criterion='aic',
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False)
forecast = model.predict(7)
print("后7次预测值：")
for i in range(len(forecast)):
    print(forecast[i])

PredicValue = []
for i in range(len(dta)):
    PredicValue.append(dta[i])
for i in range(len(forecast)):
    PredicValue.append(forecast[i])
PredicValue=pd.Series(PredicValue)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(PredicValue, '*-', label='预测值')
plt.legend()
plt.show()
