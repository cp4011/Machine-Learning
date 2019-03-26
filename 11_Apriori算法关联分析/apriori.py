from numpy import *
"""频繁项集(frequent item sets):经常出现在一起的物品的集合；                       项集支持度
关联规则(association rules):暗示两种物品之间可能存在很强的关系.                 关联规则置信度
Apriori原理：能够减少计算量.其内容是: 
                若某个项集是频繁的, 那么它的子集也是频繁的;若一个项集是非频繁的, 则它的所有超集也是非频繁的
                    Apriori算法 优点：容易编码实现
                                缺点：在大数据集上可能较慢
                                适用数据类型：数值型或标称型数据
"""


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataset):
    '''
        构建初始【候选项集】的列表，即所有候选项集只包含一个元素，
        C1是大小为1的所有候选项集的集合,是不重复的frozenset集合
    '''
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))   # rozenset"，是为了冻结集合，使集合由“可变”变为 "不可变"，这样，这些集合就可以作为字典的键值, list()


"""Apriori数据集扫描:
    对数据集中的每条交易记录tran:
    对每个候选项集can:
        检查一下can是否是tran的子集:
        如果是,则增加can的计数
        对每个候选项集：
            若该候选项集的支持度不低于最小支持度, 则保留该项集
    返回所有【频繁项集列表】
"""


def scanD(dataset, Ck, min_support):
    '''
       计算Ck中的项集在数据集合D(记录或者transactions)中的支持度,
       返回满足最小支持度的项集的集合【频繁项集列表】，和所有项集支持度信息的字典。
       Ck：候选项集列表
    '''
    ssCnt = {}
    for tid in dataset:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1
    num_items = float(len(dataset))
    return_list = []
    support_data = {}
    for key in ssCnt:
        support = ssCnt[key] / num_items
        if support >= min_support:
            return_list.insert(0, key)
        support_data[key] = support             # 汇总支持度数据
    return return_list, support_data            # 返回【频繁项集列表】满足最小支持度的项集的集合，和含有所有项集支持度信息的字典
# test
# dataset = loadDataSet()
# C1 = createC1(dataset)
# list, support = scanD(dataset, C1, 0.5)
# print(list)


"""在候选项集生成过程中, 只有在前k-2个项相同时才合并, 原始频繁项集中每一项含有k-1个元素, 为了合成得到每一项大小是k的
候选项集列表,只有在前k-2项相同时,最后一项不同时,才有可能得到频繁项集.注意这里不是两两合并, 因为限制了候选项集的大小"""


def aprioriGen(Lk, k):
    """
        函数说明：输出【候选项集Ck】
        频繁项集列表： Lk
        项集元素个数： k
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):                      # 两两组合遍历
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:                        # 若两个集合的前k-2个项相同时,则将两个集合合并
                retList.append(Lk[i] | Lk[j])
    return retList                              # 返回候选项集Ck


"""Apriori算法
    当集合中项的个数大于0时：【上一个Lk为空集时退出】
        构建一个k个项组成的候选项集的列表
        检查数据以确认每个项集都是频繁的
        保留频繁项集并构建k+1项组成的候选项集的列表
"""


def apriori(dataset, minSupport = 0.5):
    C1 = createC1(dataset)
    D = list(map(set, dataset))
    L1, supportData = scanD(D, C1, minSupport)      # 单项最小支持度判断 0.5，生成L1
    L = [L1]                                        # 列表L会逐渐包含频繁项集L1,L2,L3...
    k = 2
    # 寻找频繁项集L1,L2,L3...通过while循环来完成。创建包含更大项集的更大列表,直到下一个大的项集为空
    while len(L[k-2]) > 0:                      # 当Lk为空的时候，程序返回L并退出【k-2是因为，k上次循环加了1，且它是L列表中的索引index】
        Ck = aprioriGen(L[k-2], k)              # 生成候选项集列表
        Lk, supK = scanD(D, Ck, minSupport)     # 扫描数据集，从Ck得到Lk，丢掉不满足最小支持度要求的项集
        supportData.update(supK)                # dict.update(dict2)  dict2：添加到指定字典dict里的字典，有相同的键会替换成update的值
        L.append(Lk)                            # Lk被添加到L中
        k += 1
    return L, supportData
# test
myDat = loadDataSet()
L, suppData = apriori(myDat, 0.5)
print(u"频繁项集L：", L)
print(u"所有候选项集的支持度信息：", suppData)


"""从【频繁项集中】挖掘关联规则
一条规则P➞H的可信度定义为support(P | H)/support(P)，其中“|”表示P和H的并集。可见可信度的计算是基于项集的支持度的
若某条规则不满足最小可信度要求, 则其规则的所有子集也不会满足最小可信度要求.假设规则{0,1,2} ➞ {3}并不满足最小可信度要求，
那么就知道任何左部为{0,1,2}子集的规则也不会满足最小可信度要求
"""


# 关联规则生成函数
def generateRules(L, supportData, minConf=0.7):     # 频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = []                                # 存储所有的关联规则
    for i in range(1, len(L)):                      # 只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        for freqSet in L[i]:                        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
            H1 = [frozenset([item]) for item in freqSet]    # 该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if i > 1:                               # 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:                                   # 第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)    # 调用函数2：针对项集中只有两个元素时，计算可信度
    return bigRuleList


# 计算项集中只有两个元素的可信度：计算规则的可信度以及找到满足最小可信度要求的规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):    # 针对项集中只有两个元素时，计算可信度
    prunedH = []                                            # 返回一个满足最小可信度要求的规则列表
    for conseq in H:                                        # 后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq]     # 可信度计算，结合支持度数据
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)     # 如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            brl.append((freqSet-conseq, conseq, conf))              # 添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)                                  # 同样需要放入列表到后面检查
    return prunedH                                                  # 返回一个满足最小可信度要求的规则列表


# 生成候选规则集合【从最初的某个项集中生成更多的关联规则】
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):     # 参数:某个频繁项集,另一个是可以出现在规则右部的元素列表H
    m = len(H[0])                                                   # 计算H中的频繁项集大小m
    # 查看频繁项集频繁项集freqSet是否大到可以移除大小为m的子集
    if len(freqSet) > (m + 1):                                      # 频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m+1)                                   # 存在不同顺序，元素相同的集合，使用aprioriGen()来生成H中元素的无重复组合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)   # 计算可信度
        if len(Hmp1) > 1:                                           # 满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# test
# myDat = loadDataSet()
# L, suppData = apriori(myDat, 0.5)
# rules = generateRules(L, suppData, minConf=0.7)
# print('rules:\n', rules)


