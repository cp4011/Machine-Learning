from numpy import *


def load_dataset(filename):
    length = len(open(filename).readline().split('\t'))
    data_matrix = []
    label_matrix = []
    fr = open(filename)
    for line in fr.readlines():
        line_array = line.strip().split('\t')
        cur_line = []
        for i in range(length - 1):
            cur_line.append(float(line_array[i]))
        data_matrix.append(cur_line)
        label_matrix.append(float(line_array[-1]))
    return data_matrix, label_matrix


# 标准回归函数（求回归系数ws=(X.T * X)−1 *X.T * y）【当矩阵可逆时 该方程可用，即行列式不为0】
def stand_regeres(x_array, y_array):
    x_matrix = mat(x_array)
    y_matrix = mat(y_array).T                   # 转置成 列
    xTx = x_matrix.T * x_matrix
    if linalg.det(xTx) == 0.0:                  # 矩阵行列式|A|=0,则矩阵不可逆
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * x_matrix.T * y_matrix
    return ws                                   # 回归系数向量
# 测试
# x_arr, y_arr = load_dataset('ex0.txt')        # 输入的x0默认为1
# print(stand_regeres(x_arr, y_arr))


"""线性回归的一个问题是有可能出现欠拟合现象，因为它求的是具有最小均方误差的无偏估计 。显而易见，如果模型欠拟合将不能取得
最好的预测效果。所以有些方法允许在估计中引人一些偏差，从而降低预测的均方误差。其中的一个方法是局部加权线性回归
（Locally Weighted Linear Regression, LWLR )。在该算法中 ，我们给 待预测点附近 的每个点赋予一定的权重W；在这个子集上基于
最小均方差来进行普通的回归。与kNN一样, 这种算法每次预测均需要事先选取出对应的数据子集。该算法解出回归系数w的形式如下： 
w=(X.T * W * X)−1 *X.T * W * y
"""


# 局部加权线性回归 LWLR （局部是指 测试点附近的局部点，赋有不为0的权重）
def lwlr(testPoint, xArr, yArr, k=1.0):                       # k值越小，仅有很少的局部点将被用于训练回归模型
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))                                   # 创建对角矩阵（权重矩阵是个方阵，阶数=样本点个数)
    for j in range(m):
        diffMat = testPoint - xMat[j, :]                      # 计算样本点与预测值的距离（可得与预测点之间的距离）
        weights[j, j] = exp(diffMat*diffMat.T/(-2.0 * k**2))  # 计算高斯核函数的权重矩阵W：权重值大小随着样本点与待遇测点距离的递增，以指数级衰减
    xTx = xMat.T * (weights * xMat)                           # 如k=0.01，衰减系数k很小，此时 仅有很少的测试点附近的局部点被用于训练回归模型
    if linalg.det(xTx) == 0.0:                                # 判断是否可逆
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):            # loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def lwlrTestPlot(xArr, yArr, k=1.0):
    import matplotlib.pyplot as plt
    yHat = zeros(shape(yArr))
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i[1] for i in xArr], yArr, 'ro')
    ax.plot(xCopy, yHat)
    plt.show()
# test
# xArr, yArr = load_dataset('ex0.txt')
# print(lwlr(xArr[0], xArr, yArr, 1.0))
# lwlrTestPlot(xArr, yArr)


def rssError(yArr, yHatArr):        # 计算预测误差（参数为 数组）
    return((yArr-yHatArr)**2).sum()

# test
# abX, abY = load_dataset('abalone.txt')
# yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], k=0.1)
# yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], k=1)
# yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], k=10)
# print(rssError(abY[0:99], yHat01.T))
# print(rssError(abY[0:99], yHat1.T))
# print(rssError(abY[0:99], yHat10.T))


"""如果数据的特征比样本点还多应该怎么办？是否还可以使用线性回归和之前的方法来做预测？答案是否定的，因为在计算(XTX)−1的
时候会出错.岭回归就是在矩阵XTX上加一个λI从而使得矩阵非奇异，进而能对XTX+λI求逆。其中矩阵I是一个m∗m的单位矩阵，对角线上
元素全为1 ,其他元素全为0。而λ是一个用户定义的数值，后面会做介绍。在这种情况下，回归系数的计算公式将变成： 
w=(XTX+λI)−1XTy
岭回归最先用来处理特征数多于样本数的情况，现在也用于在估计中加人偏差，从而得到更好的估计。这里通过引入λ来限制了所有w
之和，通过引人该惩罚项，能够减少不重要的参数（缩减）
"""


# 缩减系数之“岭”回归（求回归系数）
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:                    # 如果lam设定为0的时候一样可能会产生错误，所以这里仍需要做一个检查
        print("This matrix is singular,cannot do inverse")
        return
    ws = denom.I*(xMat.T*yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)                           # 数据标准化（特征标准化处理），减去均值，除以方差
    yMat = yMat-yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))      # wMat表示岭回归系数矩阵,初始化为(30,n)维零数组
    for i in range(numTestPts):                     # 循环30次，根据不同的lambda填充ws
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat
# test
# xArr, yArr = load_dataset('abalone.txt')
# ridgrWeights = ridgeTest(xArr, yArr)
# print(ridgrWeights)
# import matplotlib.pyplot as plt
# ax = plt.figure()
# ax = ax.add_subplot(111)
# ax.plot(ridgrWeights)
# plt.show()


# 数据标准化处理（列）
def regularize(xMat):                   # 所有特征（列）按照0均值 方差为1进行标准化除了。减去各自均值，并除于方差
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)            # calc mean then subtract it off
    inVar = var(inMat, 0)               # calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


"""前向逐步线性回归
一种贪心算法，即每一步都尽可能减少误差。一开始所有的权重都设为1，然后每一步所做的决策是对某个权重增加或减少一个很小的值。
伪代码如下: 
数据标准化，使其分布满足0均值和单位方差 
在每轮迭代过程中: 
 设置当前最小误差lowestError为正无穷 
 对每个特征： 
  增大或缩小： 
    改变一个系数得到一个新的W 
    计算新W下的误差 
    如果误差Error小于当前最小误差lowestError:设置Wbest等于当前的W 
 将W设置为新的Wbest
"""


# 前向逐步线性回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat-yMean                           # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))               # 返回所有迭代中ws的变化情况，初始化为(numIt,n)维矩阵
    ws = zeros((n, 1))                          # 回归系数ws初始化为(n,1)维零数组
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):                      # 迭代numIt次，每次迭代，循环n*2次(每个特征有增大和减小两个方向)，
        print(ws.T)                             # 找出令rssError最小的方向(哪个特征，对应增大还是减小),保存ws,下次迭代在ws基础上做更新
        lowestError = inf                       # Numpy 中的无穷大
        for j in range(n):                      # 对每列特征
            for sign in [-1, 1]:                # 两次循环，计算增加或者减少该特征对误差的影响
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)        # 平方误差
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat                         # 返回所有迭代中ws的变化情况，(numIt,n)维矩阵

# test
# xArr, yArr = load_dataset('abalone.txt')
# print(stageWise(xArr, yArr, 0.01, 200))