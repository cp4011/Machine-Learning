from numpy import *
def loadSimpData():
    datMat = matrix([[1. , 2.1],
                    [2. ,  1.1],
                    [1.3,  1. ],
                    [1. ,  1. ],
                    [2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


'''构建单层决策树(decision stump)
将最小错误率minEroor设为无穷大
对数据集中的每一个特征（第一层循环）：
    对每个步长（第二层循环）：
        对每个不等号（第三层循环）：
            建立一颗单层决策树并利用加权数据集对它进行测试
            如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
返回最佳单层决策树(DS)
'''


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):        # 下一个函数每一次循环都调用，经常调用，单独拿出来写
    """
    Function：   通过阈值比较对数据进行分类【阈值一边的数据会被分到类别-1，另一边的数据会被分到类别+1】
    Input：      dataMatrix：数据集
                dimen：数据集列数
                threshVal：阈值
                threshIneq：比较方式：lt，gt
    Output： retArray：分类结果
    """
    retArray = ones((shape(dataMatrix)[0], 1))                  # 新建一个列向量用于存放分类结果，初始化都为 +1
    if threshIneq == 'lt':                                      # lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到retArray
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0      # 通过 数组过滤 来实现阈值另一边的元素都赋值为 -1
    else:                                                       # 'gt'
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray                                             # 返回分类结果


def buildStump(dataArr, classLabels, D):
    """
    Function：   找到最低错误率的单层决策树
    Input：      dataArr：数据集
                classLabels：数据标签
                D：权重向量（样本集中每个样本点具有一个权重）
    Output： bestStump：分类结果
            minError：最小错误率
            bestClasEst：最佳单层决策树
    """
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T        # 初始化数据集和数据标签
    m, n = shape(dataMatrix)                                        # 获取行列值
    numSteps = 10.0                                                 # 初始化步数，用于在特征的所有可能值上进行遍历
    bestStump = {}                              # 初始化字典，用于存储给定样本的权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m, 1)))            # 初始化类别估计值
    minError = inf                              # 将最小错误率设无穷大，之后用于寻找可能的最小错误率
    for i in range(n):                          # 遍历数据集中每一个特征
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()        # 获取数据集的最大最小值
        stepSize = (rangeMax - rangeMin) / numSteps                                 # 根据步数求得步长
        for j in range(-1, int(numSteps) + 1):                                      # 遍历每个步长
            for inequal in ['lt', 'gt']:                                            # 遍历每个不等号
                threshVal = (rangeMin + float(j) * stepSize)                        # 设定阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)    # 通过阈值比较对数据进行分类
                errArr = mat(ones((m, 1)))                                          # 初始化错误计数向量
                errArr[predictedVals == labelMat] = 0                               # 如果预测结果和标签相同，则相应位置0
                weightedError = D.T * errArr                                        # 计算权值误差，这就是AdaBoost和分类器交互的地方
                # 打印输出所有的值
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:            # 如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst             # 返回最佳单层决策树，最小错误率，类别估计值


'''AdaBoost算法伪代码：
对每次迭代：
    利用buildStump()函数找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量D
    更新累计类别估计值
    如果错误率等于0.0，则退出循环
'''


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """
    Function：   找到多个弱分类器组成的单层决策树数组（初已经训练到0错误率时，默认为训练到40个弱分类器）
    Input：      dataArr：数据集
                classLabels：数据标签
                numIt：迭代次数
    Output： weakClassArr：单层决策树列表
            aggClassEst：类别估计值
    """
    weakClassArr = []           # 初始化列表，用来存放单层决策树的信息【多个弱分类器组成的数组】
    m = shape(dataArr)[0]       # 获取数据集行数
    D = mat(ones((m, 1))/m)     # 初始化权重向量D的每个值均为1/m，D包含每个数据点的权重，权重相加为概率1
    aggClassEst = mat(zeros((m, 1)))        # 初始化列向量，记录每个数据点的类别估计累计值
    for i in range(numIt):                  # 开始迭代，如果中途不出现错误率为0的情况，则训练到40个弱分类器的时候，就结束了
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)    # 利用buildStump()函数找到最佳的单层决策树
        # print("D: ", D.T)
        # 根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))         # alpha定义为 0.5*ln[(1-错误率）/错误率]
        bestStump['alpha'] = alpha              # 保存alpha的值
        weakClassArr.append(bestStump)          # 填入数据到列表
        # print("classEst: ", classEst.T)       # classEst中分类错误的为1.分类正确所在的元素是0
        # 【为下一次迭代计算D（根据公式）】
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)   # 正确分类的样本，权重e的指数应该是-alpha，错误的是alpha。
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst         # 累加类别估计值，本质是个浮点数列向量
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 计算错误率，aggClassEst本身是浮点数，需通过sign得到二分类结果
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)       # 打印
        if errorRate == 0.0:                    # 如果总错误率为0则跳出循环
            break
    return weakClassArr, aggClassEst            # 返回单层决策树列表【多个弱分类器组成的数组】,aggClassEst用于画图


def adaClassify(datToClass, classifierArr):
    """
    Function：   AdaBoost分类函数
    Input：      datToClass：待分类样例
                classifierArr：多个弱分类器组成的数组
    Output： sign(aggClassEst)：分类结果
    """
    dataMatrix = mat(datToClass)        # 初始化数据集
    m = shape(dataMatrix)[0]            # 获得待分类样例个数
    aggClassEst = mat(zeros((m, 1)))        # 构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
    for i in range(len(classifierArr)):     # 遍历每个弱分类器
        # 基于stumpClassify得到类别估计值
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst     # 累加类别估计值
        # print(aggClassEst)        # 打印aggClassEst，以便我们了解其变化情况
    return sign(aggClassEst)        # 返回分类结果的列向量，aggClassEst大于0则返回+1，否则返回-1
# test
# dataArr, labelArr = loadSimpData()
# classiferArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 30)
# print(adaClassify([0, 0], classiferArr))
# print(adaClassify([[5, 5], [0, 0]], classiferArr))


# 在一个难数据集上应用AdaBoost
def loadDataSet(fileName):
    """
    Function：   自适应数据加载函数
    Input：      fileName：文件名称
    Output：     dataMat：数据集
                labelMat：类别标签
    """
    numFeat = len(open(fileName).readline().split('\t'))        # 自动获取特征个数，这是和之前不一样的地方
    dataMat = []; labelMat = []                                 # 初始化数据集和标签列表
    fr = open(fileName)                         # 打开文件
    for line in fr.readlines():                 # 遍历每一行
        lineArr = []                            # 初始化列表，用来存储每一行的数据
        curLine = line.strip().split('\t')      # 切分文本
        for i in range(numFeat-1):              # 遍历每一个特征，某人最后一列为标签
            lineArr.append(float(curLine[i]))   # 将切分的文本全部加入行列表中
        dataMat.append(lineArr)                 # 将每个行列表加入到数据集中
        labelMat.append(float(curLine[-1]))     # 将每个标签加入标签列表中
    return dataMat, labelMat                    # 返回数据集和标签列表


# test
dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
classiferArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
# 测试已经训练好的分类器，即数组classiferArr ：多个弱分类器组成的数组
testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
prediction10 = adaClassify(testArr, classiferArr)                   # 返回一个分类结果的列向量
errArr = mat(ones((67, 1)))
err_num = errArr[prediction10 != mat(testLabelArr).T].sum()         # 【数组过滤】
print("the error numbers is %d, error rate is %.3f%%" % (err_num, (err_num / 67)*100))


# 因为可能会过拟合，这时候就要求我们选一个合适的指标来判断多少个分类器合适，比如ROC
def plotROC(pred_strengths, class_labels):
    import matplotlib.pyplot as plt                     # AUC，曲线下的面积
    cur = (1.0, 1.0)                                    # cursor  起始点
    y_sum = 0.0                                         # variable to calculate AUC
    num_pos_clas = sum(array(class_labels) == 1.0)   # 正例的数目
    y_step = 1 / float(num_pos_clas)                    # 这两个是步长
    x_step = 1 / float(len(class_labels) - num_pos_clas)
    sorted_indicies = pred_strengths.argsort()  # get sorted index, it's reverse        # 从小到大排列，再得到下标
    fig = plt.figure()
    fig.clf()                                   # 清空
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sorted_indicies.tolist()[0]:  # np对象变成list
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')                                   # 假阳率
    plt.ylabel('True positive rate')                                    # 真阳率
    plt.title('ROC curve for AdaBoost horse colic detection system')    # AdaBoost马疝病检测系统的ROC曲线
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", y_sum * x_step)              # AUC面积

# test
# dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
# classiferArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
# plotROC(aggClassEst.T, labelArr)
