# coding=gbk

'''
###4.5.2 训练算法：从词向量计算概率
#伪代码
1.计算每个类别中的文档数目
2.针对每篇训练文档：
      对每个类别：
        如果词条出现在文档中--〉增加该词条的记数值
        增加所有词条的记数值
      对每个类别：
         对每个词条：
             将该词条的数目除以总词条书得到条件概率
       返回每个类别的条件概率

#set of Word model 词集模型
将每个词的出现与否作为1个特征，
#bage of words model 词袋模型
如果1个词在文档中出现不止一次，可能意味着包含盖茨是否出现在文档中所不能表达的某种信息
'''
from numpy import *
###4.5 屏蔽侮辱性的言论
#4-1 准备数据：从文本中构建词向量
'''
output: 
词条分割后的文档集合
类别标签
'''

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 是有侮辱性的，0正常的言论
    return postingList,classVec

#不重复的词列表        
def createVocabList(dataSet):
    vocabSet = set([])  #1-创建一个空的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) #2-合并集合，并生成最终的词汇表
    return list(vocabSet)

#出现了词汇列表中的元素，则将对应值设置为1，
#input :词汇列表，文档
#putput：文档向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) #初始化一个和单词等长集合，初始化为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec 

listOPosts,listClassses=loadDataSet()
myVocabList=createVocabList(listOPosts)
setOfWords2Vec(myVocabList,listOPosts[0])
'''
P(ci|w)=(P(w|ci) P(ci))/(P(w)) 词向量的贝叶斯准则
P(w0,w1,w2,w3..|ci) ==p(w0|ci) P(w1|ci) ....

###4.5.2 训练算法：从词向量计算概率
#伪代码
1.计算每个类别中的文档数目
2.针对每篇训练文档：
      对每个类别：
        如果词条出现在文档中--〉增加该词条的记数值
        增加所有词条的记数值
      对每个类别：
         对每个词条：
             将该词条的数目除以总词条书得到条件概率
       返回每个类别的条件概率
'''
#4-2 朴素贝叶斯分类器训练函数
#input   文档矩阵，类别标签
#output  是侮辱性文档的概率向量，非侮辱性文档的概率向量，是侮辱性文档的概率
def trainNB0(trainMatrix,trainCategory):
     numTrainDocs=len(trainMatrix)
     numWords=len(trainMatrix[0])
     pAbusive=sum(trainCategory)/float(numTrainDocs) 
     #1-初始化概率 #初始化为0  zeros
     p0Num=ones(numWords)   
     p1Num=ones(numWords)   
     p0Denom=2.0
     p1Denom=2.0
     #2-向量相加
     for i in range (numTrainDocs):
         if trainCategory[i]==1:
             p1Num+=trainMatrix[i]
             p1Denom+=sum(trainMatrix[i])
         else:
             p0Num+=trainMatrix[i]
             p0Denom +=sum(trainMatrix[i])
     #3-对每个元素作除法
     p1Vect=log(p1Num/p1Denom)
     p0Vect= log(p0Num/p0Denom)
     return p0Vect,p1Vect,pAbusive

listOPosts,listClassses=loadDataSet()
myVocabList=createVocabList(listOPosts)
#词向量矩阵
trainmat=[]
for postinDoc in listOPosts:
    trainmat.append(setOfWords2Vec(myVocabList,postinDoc))

p0V,p1V,pAb=trainNB0(trainmat,listClassses)

#set of Word model 词集模型
#bage of words model 词袋模型

###4.5.3 测试算法：根据现实情况修改分类器
## 4-3 朴素贝叶斯分类器
#input： 分类的向量,是侮辱性文档的概率向量，非侮辱性文档的概率向量，是侮辱性文档的概率
#output：比较类别的概率，返回大概率对应的类别标签
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify* p1Vec) +log(pClass1)        #元素对应相乘
    p0=sum(vec2Classify * p0Vec) +log(1.0-pClass1)   
    if p1>p0:
        return 1
    else:
        return 0

# 测试分类器
def  testingNB():
     #数据准备和训练
     listOPosts,listClasses=loadDataSet()
     myVocabList=createVocabList(listOPosts)
     trainMat=[]
     for postinDoc in listOPosts:
       trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
     p0V,p1V,pAb=trainNB0(trainMat,listClasses)
     #测试数据1, 测试
     testEntry=['love','my','dalmation']
     thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
     print (testEntry,'classified as :' ,classifyNB(thisDoc,p0V,p1V,pAb))
     #测试数据2，测试
     testEntry=['stupid','garbage']
     thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
     print (testEntry,'classified as :' ,classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()

###4.5.4 准备数据：文档词袋模型
#词袋模型函数  bagOfwrods2VecMN替换setOfWords2Vec
#区别：当每遇到1个单词时，增加词向量中对应值，不只是将对应的数值设为1
#4-4 朴素贝叶斯词袋模型
def bagOfwrods2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocalList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>4.6 示例：使用朴素贝叶斯过滤垃圾邮件
###4.6.1 准备数据：切分文本
mySent='This book is the best book on Python or M.L. I have   ever laid eyes upon'
#切分文本字符串,
mySent.split()
#标点符号也当做是词的一部分了，使用正则表达式切分，\W* 分隔符为除了单词、数字外的任意字符串
#--分次有问题
import re
regEx = re.compile('\\W+')
listoftokens=regEx.split(mySent)

###4.6.2 测试算法：使用朴素贝叶斯进行交叉验证
#4-5 文件解析及完整的垃圾邮件测试函数
#input:大字符串
#ouput:拆分为字符串列表
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>0]

textParse(mySent)

#读出邮件，并进行训练和测试
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    #读文档
    for i in range(1,26):
        wordList=textParse(open(r"D:\example\07-Ch04-byes\email\spam\%d.txt" %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open(r"D:\example\07-Ch04-byes\email\ham\%d.txt" %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #随机选择10个文件构建训练集，选出对应的数字添加到测试集
    vocabList=createVocabList(docList)
    trainingSet=range(50)
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    #测试分类
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print ('this error rate is : ' ,float(errorCount)/len(testSet))
    
spamTest()
#计算高频词
###4.7 示例：使用朴素贝叶斯分类器从个人广告中获取区域倾向
#4-6 RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

#4-7最具表征性的词汇显示函数
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
        
