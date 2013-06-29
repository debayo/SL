# coding=utf-8
'''
perceptron duality method
Created on 2013年6月28日

@author: zzy
'''
import os
import numpy as np


# An example in that book, the training set and parameters' sizes are fixed
trainingSet = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1]])
x = trainingSet[:,0]
y = trainingSet[:,1]
gram = np.array([])
# parameter
alpa = np.zeros(3, dtype=np.int16)
w = np.zeros(2,dtype=np.int16)
b = 0
learningRate = 1


def updateParameter(i):
  global alpa,b
  alpa[i] = alpa[i] + learningRate
  b = b + learningRate * y[i]
  print alpa, b

def initGram(x1,x2,N):
  gram = np.ones((N,N), dtype=np.int16)
  for i in range(N):
    for j in range(N):
      gram[i][j] = np.dot(x1[i],x1[j])
  return gram

def misClassify(i,N):
  global gram
  res = 0
  for j in range(N):
    res += alpa[j] * y[j] * gram[j][i]
  
  res += b
  res *= y[i]

  if res <= 0:
    return True
  else:
    return False

def check (N):
  flag = False
  for i in range(N):
    if(misClassify(i,N) == True):
      flag = True
      updateParameter(i)
      
  if(flag == False):
   
    computeW(alpa,N)
    print "Reulst alpa:" + str(alpa) + " b:" + str(b) + " w:" + str(w)

    os._exit(0)
    
  flag = False
def computeW(alpa,N):
  global w

  for i in range(N):
    w = w + [alpa[i] * y[i]*item for item in x[i]]

if __name__ == '__main__':
  N = len(trainingSet)
  gram = initGram(x,x,N)
#   print 'N',N
#   print 'x:',x
#   print 'y:',y
#   print 'alpa',alpa
#   print 'gram',gram
  for i in range(1000):
    check(N)
  print 'The traning set is not linear separable'
  print trainingSet
