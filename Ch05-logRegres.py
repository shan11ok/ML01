'''
Logistic Regression Working Module
1.梯度上升法的伪代码：
每个回归系数初始化为1
重复R次：
    计算整个数据集的梯度
    使用alpha * gradient 更新回归系数的向量
返回回归系数
2.随机梯度上升法的伪代码：
所有回归系数初始化为1
对数据集中每个样本
    计算该样本的梯度
    使用alpha * gradient更新归回系数值
返回回归系数数值
3.改进的 样本随机选择，alpha动态减少机制随机梯度上升算法
对比梯度上升算法主要变化在3个方面：
1).alpha在每次迭代的时候都会调整，虽然alpha会随着迭代次数不断减小，但永远不会减小到0，这是因为公式中还存在一个常数项，必须这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影响。
2).这里通过随机选取样本来更新回归系数。这种方法将减少周期性的波动。
3).改进算法还增加了一个迭代次数作为第3个参数。如果该参数没有给定的话，算法将默认迭代150次。如果给定，那么算法将按照新的参数值进行迭代

缺失值的处理方式：
1.使用可用特征的均值来填补缺失值
2.使用热数值来填补缺失值，-1
3.忽略有缺失值的样本
4.使用相似样本的均值添补缺失值
5.使用另外的机器学习算法预测缺失值
'''
from numpy import *

###5.2 基于最优化方法的最佳回归系数确定
###5.2.2 训练算法：使用梯度上升找到最佳参数
#5-1 Logistic回归梯度上升优化算法
#读取testSet.txt逐行读取，X1 X2 标签
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open(r"D:\example\08-Ch05-logistic\testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #增加X0 为1，X1 X2
        labelMat.append(int(lineArr[2])) #Y
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度提升算法
#input：数据列表，X0 X1 X2，3*100；标签列表：Y ,1*100矩阵
#output: 输出每个系数的权重参数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix 使用numpy矩阵
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix 转置便于矩阵运算
    m,n = shape(dataMatrix)
    alpha = 0.001     #移动步长
    maxCycles = 500   #迭代次数
    weights = ones((n,1))  #初始权重值
    for k in range(maxCycles):              #heavy on matrix operations 
        h = sigmoid(dataMatrix*weights)     #matrix mult 预测值
        error = (labelMat - h)              #vector subtraction 残差
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult 梯度下降值
    return weights

dataArr,labelMat=loadDataSet()
gradAscent(dataArr,labelMat)
#5-2 画出数据集和Logistic回归最佳拟合直线函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

weights = gradAscent(dataArr,labelMat)
plotBestFit(weights.getA())

#5-3 随机梯度上升算法
#input: 数据数组，标签
#output：系数权重 数组
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))  #预测值向量
        error = classLabels[i] - h               #误差向量
        weights = weights + alpha * error * dataMatrix[i]
    return weights

type(dataArr)
stocGradAscent0(array(dataArr),labelMat)
weights = stocGradAscent0(array(dataArr),labelMat)
plotBestFit(weights)

#5-4 改进的随机梯度上升算法（样本随机选择，alpha动态减少机制）
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha 每次迭代时需要调整
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

weights = stocGradAscent1(array(dataArr),labelMat)
plotBestFit(weights)

###5.3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>示例：从疝气病症预测病马的死亡率
#使用logistic回归来预测患有疝气病的马的存货，368个样本，28个特征
###5.3.1准备数据：处理数据中的缺失值
#input：回归系数，特征向量
#output：计算sigmoid，大于0.5返回1，小于返回0
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

#用于打开训练集和测试集，并对数据进行处理
def colicTest():
    frTrain = open(r"D:\example\08-Ch05-logistic\horseColicTraining.txt");
    frTest = open(r"D:\example\08-Ch05-logistic\horseColicTest.txt")
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

#调用计算10次，取平均值
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        
multiTest()