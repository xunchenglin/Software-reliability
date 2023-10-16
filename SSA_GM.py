import numpy as np
import matplotlib.pyplot as plt
#SSA部分
# 1. create a time series: Y
Y = np.array([0,9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 98, 104, 105, 116, 149, 156, 247, 249, 250, 337])
N = len(Y)
print(N)
print(Y)


# 用于创建轨迹矩阵，公式（1）
def trajectory_matrix(X, k):
    N = len(X)
    L = N - k + 1
    out = []
    for i in range(L):
        out.append(X[i:k + i])
    out = np.asarray(out)
    return out


# 创建轨迹矩阵
# N=27, K=23, L=N-K+1
K = 23
L = N - K + 1
print(f"L={L}, K={K}")
X = trajectory_matrix(Y, K)
print(X.shape)

# 奇异值分解，注意以下numpy实现的svd已经对奇异值降序排序了，对应奇异向量也已经重新排序了。
U, s, VT = np.linalg.svd(X)

# rank of X
r = np.linalg.matrix_rank(X)
print(f"rank={r}")

# 分组，这里默认每个奇异值作为一个组，所以从0遍历到r-1，共r个分组。
Xs = []
for i in range(r):
    t = s[i] * U[:, i][:, None] @ VT[i, :][:, None].T
    Xs.append(t)
print(len(Xs))
X1 = Xs[0]
X2 = Xs[1]
X3 = Xs[2]
X4 = Xs[3]


# 用于计算矩阵反对角线的均值
def average_anti_diag(X):
    """
    Average antidiagonal elements of a 2D matrix.

    Parameters:
    ----------------------------------
    X: np.array
       2D numpy array of size
    Return:
    ---------------------------------
    out: 1D array
        1D array representing averaged antidiagonal elements of X
    """
    X = np.asarray(X)
    out = [np.mean(X[::-1, :].diagonal(i)) for i in range(-X.shape[0] + 1, X.shape[1])]

    return np.asarray(out)


# 计算各个分组的子矩阵的反对角线的均值，并得到一个子序列
Y1 = average_anti_diag(X1)
Y2 = average_anti_diag(X2)
Y3 = average_anti_diag(X3)
Y4 = average_anti_diag(X4)

# 将得到的子序列求和得到重构的时间序列,优化后的数据在Y里
Y_reconstructed = Y1 + Y2 + Y3 + Y4

# 画图：原始时间序列
#plt.plot(Y, 'o-', label='original signal');
#plt.legend();

# 画图：重构的时间序列
#plt.plot(Y_reconstructed, '*-', label='reconstructed signal');
#plt.legend();

#GM部分
class GM_1_1:
    """
    使用方法：
    1、首先对类进行实例化：GM_model = GM_1_1()   # 不传入参数
    2、使用GM下的set_model传入一个一维的list类型数据: GM_model.set_model(list1)
    3、想预测后N个数据：GM_model.predict(N)
       想获得模型某个参数或实验数据拟合值，直接访问，如：GM_model.modeling_result_arr、GM_model.argu_a...等
        想输出模型的精度评定结果：GM_model.precision_evaluation()
    """

    def __init__(self):
        self.test_data = np.array(())  # 实验数据集
        self.add_data = np.array(())  # 一次累加产生数据
        self.argu_a = 0  # 参数a
        self.argu_b = 0  # 参数b
        self.MAT_B = np.array(())  # 矩阵B
        self.MAT_Y = np.array(())  # 矩阵Y
        self.modeling_result_arr = np.array(())  # 对实验数据的拟合值
        self.P = 0  # 小误差概率
        self.C = 0  # 后验方差比值

    def set_model(self, arr: list):
        self.__acq_data(arr)
        self.__compute()
        self.__modeling_result()

    def __acq_data(self, arr: list):  # 构建并计算矩阵B和矩阵Y
        self.test_data = np.array(arr).flatten()
        add_data = list()
        sum = 0
        for i in range(len(self.test_data)):
            sum = sum + self.test_data[i]
            add_data.append(sum)
        self.add_data = np.array(add_data)
        ser = list()
        for i in range(len(self.add_data) - 1):
            temp = (-1) * ((1 / 2) * self.add_data[i] + (1 / 2) * self.add_data[i + 1])
            ser.append(temp)
        B = np.vstack((np.array(ser).flatten(), np.ones(len(ser), ).flatten()))
        self.MAT_B = np.array(B).T
        Y = np.array(self.test_data[1:])
        self.MAT_Y = np.reshape(Y, (len(Y), 1))

    def __compute(self):  # 计算灰参数 a,b
        temp_1 = np.dot(self.MAT_B.T, self.MAT_B)
        temp_2 = np.matrix(temp_1).I
        temp_3 = np.dot(np.array(temp_2), self.MAT_B.T)
        vec = np.dot(temp_3, self.MAT_Y)
        self.argu_a = vec.flatten()[0]
        self.argu_b = vec.flatten()[1]

    def __predict(self, k: int) -> float:  # 定义预测计算函数
        part_1 = 1 - pow(np.e, self.argu_a)
        part_2 = self.test_data[0] - self.argu_b / self.argu_a
        part_3 = pow(np.e, (-1) * self.argu_a * k)
        return part_1 * part_2 * part_3

    def __modeling_result(self):  # 获得对实验数据的拟合值
        ls = [self.__predict(i + 1) for i in range(len(self.test_data) - 1)]
        ls.insert(0, self.test_data[0])
        self.modeling_result_arr = np.array(ls)

    def predict(self, number: int) -> list:  # 外部预测接口，预测后指定个数的数据
        prediction = [self.__predict(i + len(self.test_data)) for i in range(number)]
        return prediction

    def precision_evaluation(self):  # 模型精度评定函数
        error = [
            self.test_data[i] - self.modeling_result_arr[i]
            for i in range(len(self.test_data))
        ]
        aver_error = sum(error) / len(error)
        aver_test_data = np.sum(self.test_data) / len(self.test_data)
        temp1 = 0
        temp2 = 0
        for i in range(len(error)):
            temp1 = temp1 + pow(self.test_data[i] - aver_test_data, 2)
            temp2 = temp2 + pow(error[i] - aver_error, 2)
        square_S_1 = temp1 / len(self.test_data)
        square_S_2 = temp2 / len(error)
        self.C = np.sqrt(square_S_2) / np.sqrt(square_S_1)
        ls = [i
              for i in range(len(error))
              if np.abs(error[i] - aver_error) < (0.6745 * np.sqrt(square_S_1))
              ]
        self.P = len(ls) / len(error)
        print("精度指标P,C值为：", self.P, self.C)

    def plot(self):
        plt.figure()
        plt.plot(self.test_data, marker='*', c='b', label='row value')
        plt.plot(self.modeling_result_arr, marker='^', c='r', label='fit value')
        plt.legend()
        plt.grid()
        return plt


if __name__ == "__main__":
    GM = GM_1_1()
    # 前27次的失效时间
    GM.set_model(Y_reconstructed)
    print("模型拟合数据为：", GM.modeling_result_arr)
    GM.precision_evaluation()
    ans1=GM.predict(7)
    print("预测值为：")
    for i in range(len(ans1)):
        print(ans1[i])
    p = GM.plot()
    p.show()
