"""数据建模问题:
决策树(贪心算法)，不断将数据切分成小数据集，知道所有目标变量完全相同，或者数据不能再切分为止。
构建算法是ID3，每次选取当前最佳特征来分割数据，并且按照这个特征的所有可能取值来划分，一旦切分完成，这个特征在之后的执行
过程中不会再有任何用处。这种方法切分过于迅速，并且需要将连续型数据离散化后才能处理，这样就破坏了连续变量的内在性质。

二元切分法是另一种树构建算法，每次将数据集切分成两半，如果数据的某个特征满足这个切分的条件，就将这些数据放入左子树，
否则右子树。CART（Classification And Regression Trees，分类回归树）使用二元切分来处理连续型变量，并用总方差取代香农熵来分析模型的效果。
    优点： 可以对复杂和非线性的数据建模
    缺点： 结果不易理解                      决策树易于理解，模型树的可解释性由于回归树
    使用数据类型： 数值型和标称型数据

"""
from numpy import *


# 载入数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))         # 将每行映射成浮点数，python3返回值改变，所以需要list()
        dataMat.append(fltLine)
    return dataMat


# 切分数据集为两个子集
def binSplitDataSet(dataSet, feature, value):       # 数据集  某一特定的待切分特征   特征值
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# Tree结点类型：回归树
def regLeaf(dataSet):                                   # 生成叶结点，在回归树中是目标变量特征的均值
    return mean(dataSet[:, -1])


# 误差计算函数：回归误差
def regErr(dataSet):                                    # 计算总误差（也可以先计算出均值，然后计算每个差值再平方）
    return var(dataSet[:, -1]) * shape(dataSet)[0]      # 均方误差var()*总样本数


""" 对每个特征：
        对每个特征值：
            将数据切分成两份
            计算切分的误差
            如果当前误差小于当前最小误差，那么将当前切分设定为最佳切分并更新最小误差
    返回最佳切分的特征和阈值
    
"""


# 二元切分:找到数据的最佳二元切分方式【切分后达到最低误差】。若找到一个好的切分方式，则返回特征编号和切分特征值；有3种情况不回切分，而是直接创建叶节点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]               # 允许的误差下降值      # 切分特征的参数阈值，用户初始设置好，用于控制函数的停止时机
    tolN = ops[1]               # 切分的最小样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:     # 若所有特征值都相同，停止切分。倒数第一列转化成list且不重复
        return None, leafType(dataSet)                  # 如果剩余特征数为1，停止切分【1】。
    # 找不到好的切分特征，调用regLeaf直接生成叶结点
    m, n = shape(dataSet)
    S = errType(dataSet)                                # 最好的特征通过计算平均误差
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):                        # 遍历数据的每个属性特征
        # for splitVal in set(dataSet[:,featIndex])     # python3报错修改为下面
        for splitVal in set((dataSet[:, featIndex].T.tolist())[0]):      # 遍历每个特征里不同的特征值(矩阵某一列转置后又转为列表，此时列表有两层大括号，取[0])
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)   # 对每个特征进行二元分类
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:                                             # 更新为误差最小的特征
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:                                      # 如果切分后误差效果下降不大，则取消切分，直接创建叶结点
        return None, leafType(dataSet)                          # 停止切分【2】
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):      # 判断切分后子集大小，小于最小允许样本数停止切分【3】
        return None, leafType(dataSet)
    return bestIndex, bestValue                                 # 返回最佳切分的特征编号和阈值


"""使用字典存储树的数据结构，每个节点包含以下四个元素：待切分的特征、待切分的特征值、左子树、右子树。
创建树的代码可以重用，伪代码大致如下。
    找到最佳的待切分特征：
    如果该节点不能再分，将该节点存为叶节点
    执行二元切分
    在左子树递归调用crateTree()方法
    在右子树递归调用crateTree()方法
首先chooseBestSplit找到最好的二元切分，随后binSplitDataSet通过数组过滤切分数据集，createTree递归建立树，输入参数决定树的类型，
leafType给出建立叶节点的函数，因此该参数也决定了要建立的是模型树还是回归树，errType代表误差计算函数，ops是一个包含树构建所需的其他参数的元组
"""


# 构建tree
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 数据集默认NumPy Mat 其他可选参数【结点类型：回归树，误差计算函数，ops包含树构建所需的其他元组】
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)    # 找到最佳的待切分特征
    if feat is None:                                                # 满足停止条件时返回叶结点值
        return val
    retTree = dict()                                                # 切分后赋值
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)                # 切分后的左右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# test
# myDat = mat(loadDataSet('ex00.txt'))
# print(createTree(myDat))
# myDat1 = mat(loadDataSet('ex0.txt'))
# print(createTree(myDat1))

# 画图（图1 + 图2）
# import matplotlib.pyplot as plt
# # myDat1 = mat(loadDataSet('ex00.txt'))
# # plt.plot(myDat1[:, 0], myDat1[:, 1], 'ro')
# myDat2 = mat(loadDataSet('ex0.txt'))
# plt.plot(myDat2[:, 1], myDat2[:, 2], 'ro')
# plt.show()


"""树剪枝
如果树节点过多，则该模型可能对数据过拟合，通过降低决策树的复杂度来避免过拟合的过程称为剪枝。在上面函数chooseBestSplit中
的三个提前终止条件是“预剪枝”操作，另一种形式的剪枝需要使用测试集和训练集，称作“后剪枝”。使用后剪枝方法需要将数据集
交叉验证，首先给定参数，使得构建出的树足够复杂，之后从上而下找到叶节点，判断合并两个叶节点是否能够取得更好的测试误差，
如果是就合并
"""


# 判断输入是否为一棵树
def isTree(obj):
    return type(obj).__name__ == 'dict'         # 判断为字典类型返回true


# 返回树的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])      # 递归函数，从上往下遍历树直到找到叶节点为止，如果找到叶节点则计算它们的平均值
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


""" 基于已有的树切分测试数据：
        如果存在任一子集是一棵树，则在该子集递归剪枝过程
        计算将当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并会降低误差的话，就将叶节点合并
"""


# 树的后剪枝
def prune(tree, testData):                                  # 待剪枝的树和剪枝所需的测试数据
    if shape(testData)[0] == 0:
        return getMean(tree)                                # 确认数据集非空
    # 假设发生过拟合，采用测试数据对树进行剪枝
    if isTree(tree['right']) or isTree(tree['left']):       # 左右子树非空
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 剪枝后判断是否还是有子树
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))           # 判断是否merge
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:                       # 如果合并后误差变小
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

# 创建所有可能中最大的树
# myMat2 = mat(loadDataSet('ex2.txt'))
# myTree = createTree(myMat2, ops=(0, 1))
# # 导入测试数据，执行剪枝过程
# myMat2Test = mat(loadDataSet('ex2test.txt'))
# prune(myTree, myMat2Test)
# print(myTree)


"""模型树 
采用树结构对数据建模，除了将叶节点设定为常数，也可将其设为分段线性函数
"""


# 模型树
def linearSolve(dataSet):                   # 将数据集格式化为X Y
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:              # X Y用于简单线性回归，需要判断矩阵可逆
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):                     # 不需要切分时生成模型树叶节点
    ws, X, Y = linearSolve(dataSet)
    return ws                               # 返回回归系数


def modelErr(dataSet):                      # 用来计算误差找到最佳切分
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

# test
# myMat2 = mat(loadDataSet(('exp2.txt')))
# myTree = createTree(myMat2, modelLeaf, modelErr, (1, 10))
# print(myTree)


"""树回归和标准回归的比较 
函数treeForeCast自顶向下遍历整棵树，直到命中叶节点为止。一旦到达叶节点，它会在输入数据上调用modelEval，该参数默认值是
regTreeEval。要对回归树叶节点预测，就调用regTreeEval，要对模型树节点预测，调用modelTreeEval
"""


# 用树回归进行预测
# 1-回归树
def regTreeEval(model, inDat):
    return float(model)


# 2-模型树
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


# 对于输入的单个数据点，treeForeCast返回一个预测值。
def treeForeCast(tree, inData, modelEval=regTreeEval):      # 指定树类型
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):                            # 有左子树 递归进入子树
            return treeForeCast(tree['left'], inData, modelEval)
        else:                                               # 不存在子树 返回叶节点
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 对数据进行树结构建模
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)        # 列
    return yHat


"""树回归与标准回归比较（相关系数 越接近1.0 越好）"""

# # 1.创建回归树
# trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
# testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
# myTree1 = createTree(trainMat, ops=(1, 20))
# yHat1 = createForeCast(myTree1, testMat[:, 0])
# print(corrcoef(yHat1, testMat[:, 1], rowvar=0)[0, 1])
#
# # 2.创建一颗模型树【模型树误差更小（更加接近1.0）】
# myTree2 = createTree(trainMat, modelLeaf, modelErr, (1, 20))
# yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
# print(corrcoef(yHat2, testMat[:, 1], rowvar=0)[0, 1])
#
# # 3.标准线性回归
# ws, X, Y = linearSolve(trainMat)
# print(ws)
# m = shape(testMat)[0]
# yHat3 = mat(zeros((m, 1)))
# for i in range(m):
#     yHat3[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
# print(corrcoef(yHat3, testMat[:, 1], rowvar=0)[0, 1])
