# coding=gbk
#（从文件中）加载数据集
'''
SVM 支持向量机 support vector machines
思路：
1.平面去划分数据，平面，靠近面最近的点到面的距离
2.最大化这个 点到面的距离
3.引入拉格朗日乘子，基于约束条件优化目标函数
4.基于假设：数据100%线性可分，引入松弛变量，更新约束条件
5.使用SMO算法来训练SVM，
SMO：Sequential Minimal Optimization 序列最小优化
原理：每次循环选择两个alpha值，找到一对合适alpha，增大其中一个，减小另一个；
条件1：两个alpha必须在间隔边界之外；条件2：两个alpha还没有进行过曲建华处理或者不在边界上
SMO的伪代码：
创建一个alpha向量并将其初始化
当迭代次数小于最大迭代次数时（外循环）
 对于数据集中的每个数据向量（内循环）：
    如果该数据向量可以被优化：
       随机选择另外一个数据向量
       同时优化这两个向量
       如果两个向量都不能被优化，退出内循环
    如果所有的向量都没有被优化，增加迭代数目，继续下一次循环
'''
from numpy import *
import numpy as np
from time import sleep
####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>6.3.2 应用简化版SMO算法处理小规模数据集
#6-1 SMO算法中的辅助函数
#导入数据，对文件逐行解析，
#input：txt文件
#output：整个数据急诊，每行类标签
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr =line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return  dataMat,labelMat

#在某个区间范围内随机选择一个整数 
#input： i 是第一个alpha下标，m是所有alpha的数目
#output： j 随机输出的alpha下标
def selectJrand(i,m):
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

#调整数据，用于调整大于H，或小于L的alpha值
#input：aj  alpha值，H 上限值，L下限值
#output： 调整后的alpha值
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

dataArr,labelArr = loadDataSet(r"D:\example\09-Ch06-SVM\testSet.txt") #标签为 -1 和+1
#创建一个alpha向量并将其初始化
#当迭代次数小于最大迭代次数时（外循环）
#  对于数据集中的每个数据向量（内循环）：
#     如果该数据向量可以被优化：
#        随机选择另外一个数据向量
#        同时优化这两个向量
#        如果两个向量都不能被优化，退出内循环
#     如果所有的向量都没有被优化，增加迭代数目，继续下一次循环

 
#6-2 简化版SMO算法
#input:
#@dataMatIn 数据集
#@classLabels, 类别标签
#@C  ,常数C
#@toler, 容错率
#@maxIter 
#output:
# 常数b值，类似逻辑回归的截距
# alpha矩阵
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #随机选择第二个alpha
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                #保证alpha在0和C之间 
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print ("L==H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print ("eta>=0"); continue

                #对i进行修改，修改量和j相同，但方向相反 
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #
                #设置常数
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas

#用时9.55秒
b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
b
#大于0的alpha元素，数据过滤的用法
alphas[alphas>0]
#支持向量的个数
shape(alphas[alphas>0])
#了解哪些是支持向量的数据点
for i in range(100):
    if alphas[i]>0.0:
        print(dataArr[i] , labelArr[i])



####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>6.4利用完整Platt SMO算法加速优化
#6-3 完整版Platt SMO的支持函数
# 存储输入的参数，数据结构中包含全局使用的变量和输出要用的数据
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 误差缓存,第一列为是否有效标志位，第二列为实际的Ｅ值
        self.eCache = np.mat(np.zeros((self.m, 2))) 


# 计算并返回 E 值
def calcEk(oS, k):
    # 预测值
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    # 误差值
    Ek = fXk - float(oS.labelMat[k])
    return Ek      
  
# 内循环中的启发式方法
# 用于选择第二个 alpha 或者说内循环的　alpha 值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


#在所有的alpha都已经改变之后，更新所有的alpha
# 计算误差值并存入缓存中，在对alpha值进行优化之后会用到这个值
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

    
#6-4完整Platt SMO算法中的优化例程
#寻找决策边界的优化过程
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
       ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0) ):
            # 用于选择第二个 alpha 或者说内循环的　alpha 值
            j, Ej = selectJ(i, oS, Ei)
            alphaIoId = oS.alphas[i].copy()
            alphaJoId = oS.alphas[j].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H:
                # print('L==H')
                return 0
            
            eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
            if eta >= 0:
                # print('eta >= 0')
                return 0
            
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
            # 更新误差缓存
            updateEk(oS, j)
            
            if (abs(oS.alphas[j] - alphaJoId) < 0.00001):
                # print('j not moving enough')
                return 0
            
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJoId - oS.alphas[j])
            # 更新误差缓存
            updateEk(oS, i)
            
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIoId) * \
                oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] * \
                (oS.alphas[j] - alphaJoId) * oS.X[i,:] * oS.X[j,:].T
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIoId) * \
                oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] * \
                (oS.alphas[j] - alphaJoId) * oS.X[j,:] * oS.X[j,:].T
                
            if (0 < oS.alphas[i]) and (oS.alphas[i] < oS.C):
                oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.alphas[j] < oS.C):
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
    else:
        return 0

#6-5 完整版的SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):  # 遍历所有的值
                alphaPairsChanged += innerL(i, oS)
                # print('fullSet, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 遍历非边界值
            nonBoundIs = np.nonzero((0 < oS.alphas.A) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print('non-bound, iter: %d i: %d, pairs changed %d' %(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        # print('iteration number: %d' % iter)
    return oS.b, oS.alphas

#0.49秒
b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
alphas[alphas>0]

#计算超平面,wWWEI
# input: alpha， 数据集，标签集 
# output：w  向量
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr);
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

ws=calcWs(alphas,dataArr,labelArr)
datMat=mat(dataArr)
#计算第一个数据点的分类，<0属于-1类，>0属于+1类
datMat[0]*mat(ws)+b
#检验第一个数据点分类
labelArr[0]

# 画出完整分类图
import matplotlib.pyplot as plt 

def plotFigure(weights, b):
    x, y = loadDataSet(r"D:\example\09-Ch06-SVM\testSet.txt") 
    xarr = np.array(x)
    n = np.shape(x)[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in np.arange(n):
        if int(y[i]) == 1:
            x1.append(xarr[i,0]); y1.append(xarr[i,1])
        else:
            x2.append(xarr[i,0]); y2.append(xarr[i,1])
    
    plt.scatter(x1, y1, s = 30, c = 'r', marker = 's')
    plt.scatter(x2, y2, s = 30, c = 'g')
    
    # 画出 SVM 分类直线
    xx = np.arange(0, 10, 0.1) 
    # 由分类直线 weights[0] * xx + weights[1] * yy1 + b = 0 易得下式
    yy1 = (-weights[0] * xx - b) / weights[1]
    # 由分类直线 weights[0] * xx + weights[1] * yy2 + b + 1 = 0 易得下式
    yy2 = (-weights[0] * xx - b - 1) / weights[1]
    # 由分类直线 weights[0] * xx + weights[1] * yy3 + b - 1 = 0 易得下式
    yy3 = (-weights[0] * xx - b + 1) / weights[1]
    plt.plot(xx, yy1.T)
    plt.plot(xx, yy2.T)
    plt.plot(xx, yy3.T)
    
    # 画出支持向量点
    for i in range(n):
        if alphas[i] > 0.0:
            plt.scatter(xarr[i,0], xarr[i,1], s = 150, c = 'none', alpha = 0.7, linewidth = 1.5, edgecolor = 'red')

    plt.xlim((-2, 12))
    plt.ylim((-8, 6))
    plt.show()

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet(r"D:\example\09-Ch06-SVM\testSet.txt") 
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40) 
    w = calcWs(alphas, dataArr, labelArr)
    plotFigure(w, b)
    print(b)
    print(alphas[alphas > 0]) # 支持向量对应的 alpha > 0
    print(w)
    
###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>6.5 在复杂数据上应用核函数
#6-6 核转换函数
#分割，变成更高维度，或者计算kernel
def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T 
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) 
    else: raise NameError(' That Kernel is not recognized')
    return K


# 存储输入的参数，数据结构中包含全局使用的变量和输出要用的数据
# 引入新变量kTup，一个包含核函数信息的元祖
class optStruct:
    #初始化数据
    def __init__(self,dataMatIn, classLabels, C, toler, kTup): 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

#6-5 完整版的SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):
    #初始化输入的部分
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        #遍历全部
        if entireSet:   
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else: #遍历非边界的值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        #是否切换到遍历整个数据集
        if entireSet: entireSet = False 
        elif (alphaPairsChanged == 0): entireSet = True  
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

#6-7完整Platt SMO算法中的优化例程，使用核函数时需要对innerL()及calcEk()函数进行修改
#寻找决策边界的优化过程
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
    

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        #替换了SelectJ 
        j,Ej = selectJ(i, oS, Ei) 
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] ###
        if eta >= 0: print ("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        #更新数据到ECache    
        updateEk(oS, j) 
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        #调整b的值
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]###
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]###
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def testRbf(k1=1.3):
    #对算法进行训练
    dataArr,labelArr = loadDataSet(r'D:\example\09-Ch06-SVM\testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    #只保留支持向量 
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    #对算法进行测试
    dataArr,labelArr = loadDataSet(r'D:\example\09-Ch06-SVM\testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print ("the test error rate is: %f" % (float(errorCount)/m))

testRbf()


###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>6.3 示例： 手写识别问题，SVM解决
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#加载图像数据
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)          
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]    
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf',10)):
    #加载训练集进行训练
    dataArr,labelArr = loadImages(r'D:\example\09-Ch06-SVM\trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    #加载测试集进行测试
    dataArr,labelArr = loadImages(r'D:\example\09-Ch06-SVM\testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print ("the test error rate is: %f" % (float(errorCount)/m))

testDigits()