from numpy import *
"""逻辑回归（线性回归模型：Sigmoid函数的输入z = w0*x0 + w1*x1 +...+ wn*xn） 【目的：训练得到最佳分类的回归系数向量W】
Logistic回归其实更常被用作在分类中。它与传统的线性回归模型相比，多了一步将输出值映射到Sigmoid函数上，进而得到一个范围
在0~1之间的数值。我们可以定义大于0.5的数据为1（正）类，而小于0.5被归为0（负）类。由于函数本身的特殊性质，它更加稳定，
并且它本质上仍然是线性模型。
优点： * 计算代价不高，容易理解和实现
缺点： * 容易欠拟合，分类精度可能不高
         sigmoid函数: 1.0/(1+exp(-s))
"""

""" 1.梯度上升法(等同于我们熟知的梯度下降法，前者是寻找最大值，后者寻找最小值)，它的基本思想是：要找到某函数的最大值，
最好的方法就是沿着该函数的梯度方向搜寻。如果函数为f，梯度记为D，a为步长，那么梯度上升法的迭代公式为：w：w+a*▽wf(w)。该
公式停止的条件是迭代次数达到某个指定值或者算法达到某个允许的误差范围。我们熟知的梯度下降法迭代公式为：w：w-a*▽wf(w)
    2 使用梯度上升法寻找最佳参数
　　假设有100个样本点，每个样本有两个特征：x1和x2.为方便考虑，我们额外添加一个x0=1，将线性函数z=wTx+b转为z=wTx
(此时向量w和x的维度均价)
"""


def load_dataset():                                 # 加载数据集
    data_matrix = []
    label_matrix = []
    fr = open('testSet.txt')                        # testSet.txt：数据集
    for line in fr.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])  # 注意三个数要加上[]，里面的数要float()一下
        label_matrix.append(line_array[2])
    return data_matrix, label_matrix                # data_matrix：数据矩阵100*3, label_matrix：类别标签矩阵1*100


def sigmoid(inX):                                   # sigmoid函数: 1.0/(1+exp(-s))
    return 1.0/(1+exp(-inX))


"""梯度上升法法伪代码
每个回归系数（权重）初始化为1
重复R次：
    计算整个数据集的梯度
    使用alpha 下gradient 更新回归系数的向量
返回回归系数
"""


# （训练算法）1：梯度上升法
def grad_ascent(data_matrix_in, label_matrix):      # data_matrix_in：数据矩阵100*3, label_matrix：类别标签矩阵1*100
    data_matrix = mat(data_matrix_in)               # 转换为numpy的矩阵，以便进行矩阵乘法
    label_matrix = mat(label_matrix).transpose()    # 转换为numpy矩阵并转置为labelMat：100*1
    m, n = data_matrix.shape                        # 获得矩阵行列数
    alpha = 0.001                                   # 初始化移动步长
    max_cycles = 500                                # #初始化迭带次数
    weights = ones((n, 1))                          # 初始化权重参数矩阵【这1列权重对应着输入矩阵数据中的一行】，初始值都为1
    for i in range(max_cycles):                     # 开始迭代计算参数
        h = sigmoid(data_matrix * weights)          # h和error是个n行的列向量，100*3 * 3*1 => 100*1
        error = label_matrix - h                    # 计算误差100*1（代价函数）
        # error = label_matrix.astype('float64') - h.astype('float64') # numpy中.astype('float64')类型转换，不写画图时会报错
        weights = weights + alpha * data_matrix.transpose() * error    # 更新参数值，注意要转置【data_matrix.transpose() * error】是梯度
    return weights                                  # 返回权重参数矩阵


"""随机梯度上升法伪代码：
所有回归系数（权重）初始化为1
对数据集中每个样本
    计算该样本的梯度
    使用alpha x gradient 更新回归系数值
返回回归系数值
随机梯度上升算法与梯度上升算法在代码上很相似，但也有一些区别：1.后者的变量h和误差error都是向量，而前者则全是数值；
2.前者没有矩阵的转换过程，所有变量的数据类型都是Numpy数组
"""


# （训练算法）2.1 随机梯度上升法：stochastic gradient ascent
def stoc_grad_ascent0(data_matrix, class_labels):           # data_matrix_in：数据矩阵100*3, label_matrix：类别标签矩阵1*100
    data_matrix = array(data_matrix)                        # 列表转化为numpy的数组，以便
    m, n = shape(data_matrix)                               # 获取数据列表大小
    weights = ones(n)                                       # 初始化权重参数矩阵，初始值都为1
    alpha = 0.01
    for i in range(m):                                      # 遍历每一行数据
        h = sigmoid(sum(data_matrix[i] * weights))          # h和error是个数值  1*3 * 1*3
        error = class_labels[i] - h                         # 计算误差
        weights = weights + alpha * data_matrix[i] * error  # 更新权重值【data_matrix[i] * error】该样本的梯度
    return weights                                          # 返回权重参数矩阵


# （训练算法）2.2 改进版随机梯度上升法
def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):  # data_matrix_in：数据矩阵100*3, label_matrix：类别标签矩阵1*100
    data_matrix = array(data_matrix)          # 将列表如[1, 2, 3] 转换成numpy数组[1 2 3]  反过来转化成列表是：array([1,2,3]).tolist()
    m, n = shape(data_matrix)                 # 数组里是同类型的（如整数），内存连续。list里元素是地址的引用（一系列指针），内存不一定连续
    weights = ones(n)
    for i in range(num_iter):                 # 开始迭代，迭代次数为numIter
        data_index = list(range(m))           # 为减少周期性波动，随机选取样本来更新参数。样本数据的索引。返回的是range对象，需list转换一下
        for j in range(m):
            alpha = 4/(1.0+i+j) + 0.0001      # alpha在每次迭代中都会调整，缓解数据波动或高频波动【常数项：为了保证不会减小到0】
            rand_index = int(random.uniform(0, len(data_index)))    # 随机产生索引 样本数据索引data_index 的下标，从而减少随机性的波动
            h = sigmoid(sum(data_matrix[rand_index] * weights))     # 序列号对应的元素与权重矩阵相乘，求和后再求sigmoid。注意sum()
            error = class_labels[rand_index] - h                    # 求误差
            weights = weights + alpha * data_matrix[rand_index] * error     # 更新权重矩阵
            del(data_index[rand_index])         # 删除该样本数据索引，data_index中少一个数了，产生rand_index的范围也会小一个
    return weights                              # 返回权重参数矩阵


def classify_vector(inX, weights):          # 分类函数  inX：计算得出的矩阵100*1，weights：权重参数矩阵
    prob = sigmoid(sum(inX * weights))      # #计算sigmoid值   记住要sum()求和
    if prob > 0.5:                          # 返回分类结果
        return 1.0
    else:
        return 0.0


def colic_test():                               # 训练和测试函数
    fr_train = open('horseColicTraining.txt')   # 打开训练集
    training_set = []                           # 初始化训练集数据列表
    training_labels = []
    for line in fr_train.readlines():           # 遍历训练集数据
        curr_line = line.strip().split('\t')    # 切分数据集
        line_array = [float(i) for i in curr_line]
        training_set.append(line_array[:-1])    # 除了最后一位
        training_labels.append(line_array[-1])
    train_weights = stoc_grad_ascent1(training_set, training_labels, 1000)      # 训练并获得权重参数

    error_count = 0
    num_test_vector = 0
    fr_test = open('horseColicTest.txt')
    for line in fr_test.readlines():            # 遍历测试集数据
        num_test_vector += 1
        curr_line = line.strip().split('\t')
        line_array = [float(i) for i in curr_line]
        if classify_vector(line_array[:-1], train_weights) != line_array[-1]:   # 如果分类结果和分类标签不符，则错误计数+1
            error_count += 1
    error_rate = (float(error_count) / num_test_vector)         # 计算分类错误率
    print("the error rate of this test is: %f" % error_rate)
    return error_rate                                           # 返回分类错误率


def muti_test():                    # 求均值函数
    num_tests = 10                  # 迭代次数
    error_sum = 0.0                 # 初始错误率和
    for i in range(num_tests):      # 调用十次colicTest()，累加错误率
        error_sum += colic_test()
    print("After %d iterations, the average error rate is: %f" % (num_tests, error_sum / num_tests))    # 打印平均分类结果
# 测试
muti_test()

