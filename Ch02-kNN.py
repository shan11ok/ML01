'''
kNN: k Nearest Neighbors
Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)        
Output:     the most popular class label
伪代码：
对未知类别属性的数据集中的每个点依次执行以下操作：
1、算距离：计算已知类别数据集中的点与当前点之间的距离
2、排序：按照距离进行排序
3、选k个点：选取与当前点距离最小的k个点
4、算频率：确定前k个点所在的类别的出现频率
5、返回前k个点出现频率最高的类别作为当前点的预测分类

示例：在约会网站上使用k-近邻算法
1、收集数据：提供文本文件
2、准备数据：使用python解析文本文件
3、分析数据：使用matplotlib画二维扩散图进行分析
4、训练算法：
5、测试算法：
6、使用算法：
@author: shan11ok
'''
from numpy import *
import operator
from os import listdir

#创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>算法核心代码 
# 2-1 k-近邻算法，计算函数
# inX要检测的数据，输入向量
# dataSet训练数据集，
# labels标签向量，结果集，labels的个数与dataSet的行数相同
# 参数k，k个数值
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #返回dataset行数
    #1-距离计算
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    #2-选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #3-排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #iteritems改为items
    return sortedClassCount[0][0]

#导入数据
group,labels=createDataSet()
#测试分类结果
a=classify0([0,0],group,labels,2)
a=classify0([0.9,0.6],group,labels,2)
a=classify0([1.2,1.3],group,labels,2)
a=classify0([0.3,0.5],group,labels,2)

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>示例1：改进约会网站的配对效果
##2.2.1准备数据：从文本文件中解析数据
'''
海伦收集到的特征数据：
1.每年获得的飞行里程数
2.玩视频游戏所耗的时间百分比
3.每周消费的冰淇淋公升数
收集了1000行数据
海伦对约会对象做分类
1.不喜欢的人didntLike
2.魅力一般的人smallDoses
3.极具魅力的人largeDoses
原始数据为datingTestSet.txt
把类型改为标签1，2，3 为datingTestSet2.txt
'''
#2-2 将文本记录到转换Numpy的解析程序，即从文件中加载数据
def file2matrix(filename):
    fr = open(filename)                         #打开文件
    #1-得到文件的行数
    numberOfLines = len(fr.readlines())         #get the number of lines in the file获取行数
    #2-创建返回numpy的矩阵
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return准备矩阵 numberOfLines行 3列
    classLabelVector = []                       #prepare labels return 准备结果标签  
    fr = open(filename)
    index = 0
    #3-解析文件数据到列表
    for line in fr.readlines():
        line = line.strip() #截取所有回车符号
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector         #返回矩阵和标签

#里程数、
group,labels=file2matrix(r"D://example//05-knn//datingTestSet2.txt")

##2.2.2 分析数据：使用matplotlib创建散点图
import matplotlib
import matplotlib.pyplot as plt
fig =plt.figure()
ax = fig.add_subplot(111)
ax.scatter(group[:,1],group[:,2])
plt.show()

#用不同的颜色标记不同的类型
ax.scatter(group[:,1],group[:,2],15.0*array(labels),15.0*array(labels))

##2.2.3 准备数据：归一化数值
#2-3归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) #最小值
    maxVals = dataSet.max(0) #最大值
    ranges = maxVals - minVals #范围
    normDataSet = zeros(shape(dataSet)) #
    m = dataSet.shape[0]    #某一维的长度
    normDataSet = dataSet - tile(minVals, (m,1))    #
    #1-特征值相除
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

#a为归一化的效果，b最大最小值的范围，c为最小值
a,b,c=autoNorm(group)

##2.2.4 测试算法：作为完整程序验证分类器
#2-4分类器针对约会网站的测试代码
#例1：约会的分类函数
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix(r"D://example//05-knn//datingTestSet2.txt")       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)

datingClassTest()

##2.2.5使用算法：构建完整可用系统  --待完善
#2-5约会网站预测函数
def classifyperson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input())

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>示例2：手写识别系统，数字识别
##2.3.1 准备数据：将图像转换为测试向量
def img2vector(filename):
    returnVect = zeros((1,1024))  #32*32为1024的长度
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])   #向量运算
    return returnVect

a=img2vector(r"D://example//05-knn//digits//testDigits//0_0.txt")

##2.3.2 测试算法：使用k-近邻算法识别手写数字
#2-6 手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(r"D:\example\05-knn\digits\trainingDigits")    #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(r"D:\example\05-knn\digits\trainingDigits\%s" % fileNameStr)
    testFileList = listdir(r"D:\example\05-knn\digits\testDigits")        #iterate through the test set
    #start test
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r"D:\example\05-knn\digits\testDigits\%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))

handwritingClassTest()
#使用代码
def handwriteiclassifier(filename):
    hwLabels = []
    trainingFileList = listdir(r"D:\example\05-knn\digits\trainingDigits")    #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(r"D:\example\05-knn\digits\trainingDigits\%s" % fileNameStr)
    #start test
    fileNameStr = filename
    vectorUnderTest = img2vector(filename)
    classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    print (classifierResult)


handwriteiclassifier(r"D:\example\05-knn\digits\testDigits\5_17.txt")
