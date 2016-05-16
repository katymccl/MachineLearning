import time
from StringIO import StringIO
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

from sklearn import grid_search
import numpy as np
import sklearn
from sklearn import svm
import sys
import random
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.svm import NuSVR
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

from sklearn import ensemble
def supportVectorMachine(trainMatrix, traintarget, testMatrix ):

	clf = svm.SVR(kernel='rbf', cache_size=1000, C=1.0, coef0=0.0, degree=3, epsilon=0.5, gamma=0.001, max_iter=-1, shrinking=True, tol=0.0001, verbose=False)
	clf.fit(trainMatrix, traintarget)
	
	predictedVals = clf.predict(testMatrix)

	for k in range(len(predictedVals)):
		print predictedVals[k]
		print "\n"



def lassoReg(trainMatrix, traintarget, testMatrix ):
	clf = linear_model.Lasso(alpha=0.0001, max_iter=10000, normalize=False, positive = False, precompute=False, random_state=None, tol=0.0001, warm_start=False, fit_intercept=True, copy_X=True, selection='cyclic')
	clf.fit(trainMatrix, traintarget)

	predictedVals = clf.predict(testMatrix)

	for k in range(len(predictedVals)):
		print predictedVals[k]
		print "\n"

def parseData():
	traintargetfile = open("proj_data/task4/train.RT",'r')
	traindatafile = open("proj_data/task4/train.sparseX",'r')

	testdatafile = open("test_features/task4/test.sparseX",'r')
	testdata = np.loadtxt(testdatafile)
	testdatafile.close()

	traindata = np.loadtxt(traindatafile)
	traintarget = np.loadtxt(traintargetfile)

	traintargetfile.close()
	traindatafile.close()

	row = []
	col = []
	val = []

	for k in range(len(traindata)):
	    i = traindata[k][0]
	    j = traindata[k][1]
	    value = traindata[k][2]
	    row.append(int(i))
	    col.append(int(j))
	    val.append(int(value))

	trainMatrix = coo_matrix((val, (row, col)), shape=(53445, 75000))
	trainMatrix = csc_matrix(trainMatrix)


	row = []
	col = []
	val = []

	for k in range(len(testdata)):
	    i = testdata[k][0]
	    j = testdata[k][1]
	    value = testdata[k][2]
	    row.append(int(i))
	    col.append(int(j))
	    val.append(int(value))

	testMatrix = coo_matrix((val, (row, col)), shape=(53969, 75000))
	testMatrix = csc_matrix(testMatrix)

	return trainMatrix, traintarget, testMatrix

def main():
	trainMatrix, traintarget, testMatrix = parseData()

	supportVectorMachine(trainMatrix, traintarget, testMatrix  )#A

	#lassoReg(trainMatrix, traintarget, testMatrix  )#B

main()
