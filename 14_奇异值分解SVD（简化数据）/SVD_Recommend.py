from numpy import *
from numpy import linalg


"""numpy.linalg.svd()
将矩阵D(m*n) 分解为3个矩阵 U(m*m) ∑(m*n) V(n*n).T    其中U、VT都是单式矩阵（unitary matrix），Σ是一个对角矩阵，对角
元素称为奇异值，它们对应了原始矩阵Data的奇异值，Σ中只有从大到小排列的对角元素。
    一般选择前几个奇异值中占总奇异值总能量90%的那些奇异值（能量：奇异值平方）
在PCA中我们根据协方差矩阵得到特征值，它们告诉我们数据集中的重要特征，Σ中的奇异值亦是如此。奇异值和特征值是有关系的，
这里的奇异值就是矩阵Data * Data^{T}特征值的平方根。

奇异值分解的优缺点：                                      
    优点：简化数据，去除噪声，提高算法的结果。
    缺点：数据的转换可能难以理解。
    使用数据类型：数值型数据。                           应用：（1）隐性语义索引（LSI/LSA） （2）推荐系统
矩阵分解
    矩阵分解是将数据矩阵分解为多个独立部分的过程。
    矩阵分解可以将原始矩阵表示成新的易于处理的形式，这种新形式是两个或多个矩阵的乘积。（类似代数中的因数分解）
    举例：如何将12分解成两个数的乘积？（1，12）、（2，6）、（3，4）都是合理的答案。
    
    基于协同过滤的推荐引擎（协同过滤是通过将用户与其他用户的数据进行对比来实现推荐的）： 1.基于物品的相似度（两个餐馆菜肴之间的距离）
2.基于用户的相似度(计算用户距离），行与行（两用户）之间的比较（两行对应坐标相减，平方，开根号）是基于用户的相似度，列与列之间的比较
是基于物品的相似度。基于物品的相似度计算时间会随着物品数量的增加而增加，基于用户的相似度计算的时间会随用户数量的增加而增加。
基于用户 相似度计算可能数量庞大。而一家店item商品最多几千件，因此可以基于物品相似度的推荐引擎。
"""


def loadExData():
    return [[0, 0, 0, 2, 2],            # 每行代表一个用户
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],            # 每列代表一个物品，如烤肉饭
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],            # 值代表评价分数
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]            # 0代表未评价

# 相似度计算 距离越近相似度越高，距离越远相似度越低


# 欧式距离相似度：范围在0-1之间                  【1/(1+欧式距离)】
def ecludeSim(inA, inB):                            # inA inB 都是列向量
    return 1.0 / 1.0 + linalg.norm(inA - inB)       # linalg.norm(计算2阶范数),距离为0,则相似度为1.0


# 余弦相似度(-1~1)：范围调整到0-1之间            【向量A与向量B 的夹角余弦 A*B / (||A||*||B||)】
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)     # 向量A的模(2阶范数) * 向量B的模(2阶范数)
    return 0.5 + 0.5 * (num /denom)                 # 范围在-1~1之间，转到0-1之间


# 皮尔逊距离相关系数：corrcoef() 皮尔逊相关系数取值范围在 -1 ~ 1之间，范围转到0-1之间
def pearsSim(inA, inB):
    if len(inA) < 3:    # 检查是否存在3个或更多的点，如果不存在，函数返回1.0，因为此时两个向量完成相关
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]        # -1 ~ 1之间，范围转到0-1之间


"""基于协同过滤的推荐引擎（协同过滤是通过将用户与其他用户的数据进行对比来实现推荐的）：基于物品的 + 基于用户的相似度，基于用户可能数量庞大
基于物品相似度的推荐引擎（推荐未尝过的菜肴）推荐系统的过程流程是：给定一个用户，系统会为此用户返回N个最好的推荐菜：
    （1）寻找用户没有评级的菜肴，即用户—物品矩阵中的0值；
    （2）在用户没有评级的所有物品中，对每个物品预计一个可能的评级分数。
    （3）对这些物品的评分从高到低进行排序，返回前N个物品。
"""


# 基于物品相似度的推荐引擎（推荐未尝过的菜肴）：列与列之间比较相似度（两列对应坐标相减，平方，开根号）
def standEst(dataMat, user, simMeas, item):     # 标准相似度方法估计商品评价:参数:数据矩阵、该用户的编号如0 Jim、相似度计算方法和该物品的编号如2寿司饭（Jim尚没尝过寿司饭）P258
    """用来计算在给定相似度计算方法条件下，该用户uesr对该物品item（此用户还未尝过的菜）的估计评分值"""
    n = shape(dataMat)[1]                       # 列维度 物品数量
    simTotal, ratSimTotal = 0.0, 0.0            # 估计评分值变量初始化
    for j in range(n):                          # 对每一个物品
        userRating = dataMat[user, j]           # 获得该用户2的评价
        if userRating == 0:    # 若用户2未评价的则跳过，直到找到此用户评价过的物品（列），将该列和想知道物品评分值item的那一列进行相似度比较
            continue           # 对两列同时大于0的计算相似度（即用户都已经尝过两列菜肴），然后累加 相似度 * 该用户2在j列物品上的评分 的乘积，最后除于总共的相似度
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]  # 寻找哪些用户都已评价这两个菜(两列物品评价值处都大于0的行)
        if len(overLap) == 0:                   # 若同时评价了两个菜的用户数量为零，则相似性为0
            similarity = 0
        else:                                   # 若存在同时评价了两个菜，计算共同评价了的评价矩阵之间的相似性
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print("the %d and %d similarity is: %f" % (item, j, similarity))
        simTotal += similarity                  # 累计相似度
        ratSimTotal += similarity * userRating  # 根据评分计算比率，选取的物品和各个物品相似度 * 用户对物品的评价权重
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal           # 最后将相似度评分的成绩进行归一化到0-5之间


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):  # 数据集矩阵 用户 推荐物品数量  相似性函数  估计方法
    """推荐引擎，会调用standEst()函数，产生最高的N个推荐结果（预测用户评级较高的前N种物品）"""
    unRatedItem = nonzero(dataMat[user, :].A == 0)[1]                   # 寻找用户未评价的物品（没有尝过的菜）
    if len(unRatedItem) == 0:
        return "you rated everything"
    itemScores = []
    for item in unRatedItem:                                            # 遍历该用户每一个未评价的菜肴
        estimatedScore = estMethod(dataMat, user, simMeas, item)        # 该未尝过的菜的预测得分
        itemScores.append((item, estimatedScore))                       # 记录该菜肴以及对应的预测评分
    return sorted(itemScores, key=lambda x: x[1], reverse=True)[:N]     # 逆序排序前N个预测评价最高的未评价的菜，返回前N个未评级的菜肴

# test
# myMat = mat(loadExData())
# myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
# myMat[3, 3] = 2
# print(recommend(myMat, 2))


# 实际的菜肴评价矩阵稀疏很多，用SVD简化数据，降低数据的维数，利用SVD将所有的菜肴映射到一个低维空间中去，在低维空间中，可以利用前面相同的相似度计算方法进行推荐
def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],      # 行对应用户，列对应物品
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],      # SVD分解的关键在于，降低了user的维度，从n变到了4
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 基于矩阵奇异值分解转换的商品评价估计【利用SVD将11维的矩阵转化成4维的矩阵（4维已经达到总能力的90%以上信息）】
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]                   # 物品种类数据
    simTotal, ratSimTotal = 0.0, 0.0        # 相似性总和 变量初始化
    U, Sigma, VT = linalg.svd(dataMat)      # 数据集矩阵 SVD奇异值分解  返回的Sigma 仅为对角线上的值
    Sig4 = mat(eye(4) * Sigma[:4])          # # 前四个已经包含了90%以上的能量了，转化成对角矩阵
    # 计算能量 Sigma2 = Sigma**2  energy=sum(Sigma2)   energy90= energy*0.9   energy4 = sum(Sigma2[:4])
    xformedItems = dataMat.T * U[:, :4] * Sig4.I        # 降维：变换数据到低维空间
    for j in range(n):                                  # 计算相似度，给出归一化评分
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:                # 跳过其他未评价的商品
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)     # 计算svd分解转换过后矩阵的相似度
        print("the %d and %d similarity is: %f" % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# test
# myMat = mat(loadExData2())
# # print(recommend(myMat, 1, estMethod=svdEst))         # 默认余弦相似度
# print(recommend(myMat, 1, estMethod=svdEst, simMeas=ecludeSim))    # 欧氏距离相似度


# 打印矩阵
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for j in range(32):
            if float(inMat[i, j]) > thresh:      # 由于矩阵包含了浮点数,因此必须定义浅色和深色，通过阈值来界定
                print("1", end=" ")
            else:
                print("0", end=" ")
        print('')


# SVD实现图像压缩
def imgCompress(numSV=3, thresh=0.8):                   # 允许基于任意给定的奇异值数目来重构图像
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []                                     # 定义一个临时存储每行的列表
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = linalg.svd(myMat)                    # SVD分解
    SigRecon = mat(zeros((numSV, numSV)))               # 初始化新对角矩阵
    for k in range(numSV):                              # 构造对角矩阵，将奇异值填充到对角线
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV]*SigRecon*VT[:numSV, :]      # 通过截断的U和VT矩阵,用SigRecon来重构图像,默认U和VT都是32*3的矩阵，再加上3个奇异值
    print("****reconstructed matrix using %d singular values******" % numSV)    # 只需要存储 32*3 + 32*3 + 3 = 195 个数字
    printMat(reconMat, thresh)

#test
"""只需要存储 32*2 + 32*2 + 2 = 130个数字，与原数目1024相比，获得几乎10倍的压缩比"""
# imgCompress(2)
