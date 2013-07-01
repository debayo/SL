'''
Naive Bayes SL Book Example 4_1 4_2
Created on Jul 1, 2013

'''
import numpy as np

def trainNB(trainingSet,lamda=0):
  #two class: 1 and 0
  K=2
  N=len(trainingSet)
  dataList  = trainingSet[:,0]
  labelList = trainingSet[:,1]
  n = len(dataList[0])
  dataSet = []
  p0 = []
  p1 = []
  
  #get feature value set
  for i in range(n):
    dataSetDim = set([])
    for dataItem in dataList:
      dataSetDim.add(dataItem[i])
    dataSetDim = list(dataSetDim)
    dataSet.append(dataSetDim)
  #for debug 
#   print 'dataSet',dataSet
#   print 'dataList',dataList
#   print 'labelList',labelList
  
  #compute condition probablity 
  b1 = np.sum(labelList)
  b0 = N-b1
  for j in range(n):
    sj = len(dataSet[j])
    p0Item = np.zeros(sj,dtype=np.float32)
    p1Item = np.zeros(sj,dtype=np.float32)
    
    for i in range(N):
      flagList = np.zeros(sj,dtype=np.float32)
      flagList[dataSet[j].index(dataList[i][j])]=1.0
      if(labelList[i] == 1):
        p1Item=p1Item+flagList
      else:
        p0Item=p0Item+flagList
        
    p0Item = (p0Item+lamda)/(b0+sj*lamda)
    p1Item = (p1Item+lamda)/(b1+sj*lamda)
    p0.append(p0Item)
    p1.append(p1Item)  
    
  num0 = 1.0*(b0+lamda)/(b1+b0+K*lamda)
  num1 = 1.0*(b1+lamda)/(b1+b0+K*lamda)
  
#   print 'p1',p1
#   print 'p0', p0
  return dataSet, p1, p0, num1, num0

def testNB(testData,dataSet,p1,p0,num1,num0):
  y0 = 1.0*num0
  y1 = 1.0*num1
  for i in range(len(testData)):
    y0 *= p0[i][dataSet[i].index(testData[i])]
    y1 *= p1[i][dataSet[i].index(testData[i])]
  print 'y1',y1,'y0',y0
  
  if(y0<y1):
    str = 'classefied as 1'  # @ReservedAssignment
  else:
    str = 'classified as 0' # @ReservedAssignment
  print testData,str


def test(testData=[2,'S']):
    trainingSet = np.array([
                [[1, 'S'],0],
                [[1, 'M'],0],
                [[1, 'M'],1],
                [[1, 'S'],1],
                [[1, 'S'],0],
                [[2, 'S'],0],
                [[2, 'M'],0],
                [[2, 'M'],1],
                [[2, 'L'],1],
                [[2, 'L'],1],
                [[3, 'L'],1],
                [[3, 'M'],1],
                [[3, 'M'],1],
                [[3, 'L'],1],
                [[3, 'L'],0]
                ])
    
    dataSet, p1, p0,num1,num0 = trainNB(trainingSet,lamda=0)

    testNB(testData, dataSet, p1, p0,num1,num0)
    
    dataSet, p1, p0,num1,num0 = trainNB(trainingSet,lamda=1)
    testNB(testData, dataSet, p1, p0,num1,num0)

if __name__ == '__main__':
    test()
  
