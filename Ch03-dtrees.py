'''
dt: dection tree
Input:       
Output: 

划分数据集，直到所有具有相同类型的数据均在一个数据子集内
创建分支的伪代码：
检测数据集中的每个子项是否属于同一分类：
If so return 类标签
Else
    寻找划分数据集的最好特征
    划分数据集
    创建分支节点
        for 每个划分的子集
            调用函数createBranch并增加返回结果到分支节点中
    return 分支节点
示例：海洋生物数据
@author: shan11ok
'''
from math import log  #数学的包
import operator

##3.1.1 信息增益
#例0：海洋生物数据，鱼鉴定数据集
# dataset 
# 序号
# 特征1：不浮出水面是否可以生存
# 特征2：是否有脚蹼
# labels 标签：是否属于鱼类
def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

ds,lbs=createDataSet()
#熵越高，则混合的数据越多
#3-1计算给定数据集的香农熵，
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)  #数据集的长度
    #1-为所有可能分类创建字典
    labelCounts={}           #定义一个字典
    for featVec in dataSet:
        currentLabel=featVec[-1]  #key值即标签，数据字典的键值为数据集的最后一列
        if currentLabel not in labelCounts.keys():  #如果键值不存在，则扩展字典并将当前键值加入字典
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1 #每个键值都记录了当前类别的出现次数
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries #使用所有类别标签的发生频率计算类别出现的概率
        shannonEnt-=prob * log(prob,2) #利用这个概率计算香农熵
    return shannonEnt

sh=calcShannonEnt(ds)
ds[0][-1]='maybe'

##3.1.2 划分数据集
# split dataset划分数据集
# input: 待划分数据集，划分数据集的特征，需要返回的特征值
def splitDataSet(dataSet,axis,value):
    retDataSet=[]  #1-创建新的list对象
    for featVec in dataSet:
        if featVec[axis]==value:  
            reducedFeatVec=featVec[:axis]  #2-抽取符合特征的数据
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

splitDataSet(ds,0,1)
splitDataSet(ds,2,'maybe')

#append vs extend
a=[1,2,3]
b=[4,5,6]
a.append(b)
a
a=[1,2,3]
a.extend(b)
a

#3-3 选择最好划分数据集最好的方式
def chooseBestFeatureToSpit(dataSet):
    numFeatures=len(dataSet[0])-1 #特征的个数
    baseEntropy=calcShannonEnt(dataSet) #计算熵值
    bestInfoGain=0.0 
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet] #创建唯一的分类标签列表
        uniqueVals=set(featList)  #set类型为不重复集合列表，某一特征所有的值
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)  #某个特征为某个值的子集
            prob=len(subDataSet)/float(len(dataSet))  #计算熵值
            newEntropy+=prob * calcShannonEnt(subDataSet) #新的熵值 
        infoGain =baseEntropy-newEntropy  #信息增益
        if(infoGain>bestInfoGain):  #信息增益对比，保留信息增益最好的值，及特征
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

chooseBestFeatureToSpit(ds)

#统计每类标签出现的频率
def majorityCnt(classList):
    classCount={}
    for vote in classList:  #在列表里面
        if vote not in classCount.keys():  #是否在统计内
            classCount[vote]=0
        classCount[vote] +=1
    sortedClassCount =sorted(classCount.iteritems(),key=opertor.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


#3-4 创建数的函数代码
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  #类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  #遍历完所有特征时返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSpit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])   #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree
 

mytree=createTree(ds,lbs)
lbs
ll=['yes','yes','no','no','no']
tr=createTree(ds,ll)

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>3.2在python中使用matplotlib注解绘制树形图
#3-5 使用文本注释绘制树节点
import matplotlib.pyplot as plt

#1-定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#2-绘制到箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def createPlot():
   fig = plt.figure(1, facecolor='white')
   fig.clf()
   createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
   plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
   plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
   plt.show()

#绘制树节图
createPlot()

#3-6 获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# 输出预先存储的树信息
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


#3-7 plotTree函数
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

createPlot(mytree)

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>3.3 测试和存储分类器
##3.3.1 测试算法：使用决策树执行分类
#3-8 使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr=inputTree.keys()[0]
    secondDict =inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
    return classLabel
lbs=['yes','no']
classify(tr,lbs,[1,0])

#3-9 使用 pickle 模块存储决策树
def storeTree(inputTree ,filename):
    import pickle
    fw=open(filename,"w")
    pickle.dump(inputTree,fw)
    fw.close()

def loadTree(filename):
    import pickle
    fr=open(filename,"r")
    return pickle.load(fr)

#保存树
storeTree(tr,r"D:\example\06-Ch3-dtree\data.txt")
tr1=loadTree(r"D://example//06-Ch3-dtree//data.txt")

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>示例：使用决策树预测隐形眼镜类型
'''
年龄，视力，处方，眼睛是否干涩，

年龄  ：青少年，成年，老年
视力  ：近视，深度近视
处方  ：有，无
泪腺  ：干涩，正常

结果：不适合，硬质，软质
'''
fr=open(r"D:\example\06-Ch3-dtree\lenses.txt")
lenses=[inst.strip().split("\t") for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate'] 
lensesTree=createTree(lenses,lensesLabels)
lensesTree
createPlot(lensesTree)