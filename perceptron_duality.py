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
x = trainingSet[:, 0]
y = trainingSet[:, 1]
gram = np.array([])
# parameter
alpa = np.zeros(3, dtype=np.int16)
w = np.zeros(2, dtype=np.int16)
b = 0
learningRate = 1
N = len(trainingSet)

# update parameters using stochastic gradient descent
def updateParameter(i):
  global alpa, b, y
  alpa[i] = alpa[i] + learningRate
  b = b + learningRate * y[i]
  print alpa, b
  
# calculate the Gram matrix
def initGram():
  global alpa, b, x, y, N, w
  gram = np.ones((N, N), dtype=np.int16)
  for i in range(N):
    for j in range(N):
      gram[i][j] = np.dot(x[i], x[j])
  return gram

# judge if sample i has been misClassfied
def misClassify(i):
  global alpa, b, x, y, N, w
  res = 0
  for j in range(N):
    res += alpa[j] * y[j] * gram[j][i]
  res += b
  res *= y[i]

  if res <= 0:
    return True
  else:
    return False
  
# check if the hyperplane can classify the examples correctly
def check():
  global alpa, b, x, y, N, w
  flag = False
  for i in range(N):
    if(misClassify(i) == True):
      flag = True
      updateParameter(i)
      
  if(flag == False):
    for i in range(N):
      w = w + [alpa[i] * y[i] * item for item in x[i]]
    print "Reulst alpa:" + str(alpa) + " b:" + str(b) + " w:" + str(w)
    os._exit(0)
    

if __name__ == '__main__':
  gram = initGram()# initialize the Gram matrix
#   print 'N',N
#   print 'x:',x
#   print 'y:',y
#   print 'alpa',alpa
#   print 'gram',gram
  for i in range(1000):
    check()
  print 'The traning set is not linear separable'
  print trainingSet
