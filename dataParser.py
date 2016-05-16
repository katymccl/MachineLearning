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

def gradientBoost(devMatrix, trainMatrix, devtarget, traintarget):
	f = open('gradientBoost.log', 'a')
	f.write('Model started')
	est = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, max_depth=5,verbose=1)
	value = ('Model: gradient boost with parameters ',est.get_params(False))
	print (str(value))
	f.write(str(value))
	est.fit(trainMatrix, traintarget)
	value1 = mean_squared_error (traintarget, est.predict(trainMatrix))
	value2 = mean_squared_error (devtarget, est.predict(devMatrix))
	print 'MSE modified train'
	print value1
	f.write('MSE mod train')
	f.write(str(value1))
	f.write('MSE mod dev')
	f.write(str(value2))
	print 'MSE modified dev'
	print value2

	f.write("MSE for train: %.2f" % np.mean((clf.predict(trainMatrix) - traintarget) ** 2))
	f.write("MSE for dev: %.2f" % np.mean((clf.predict(devMatrix) - devtarget) ** 2))
	print("MSE for dev: %.2f" % np.mean((clf.predict(devMatrix) - devtarget) ** 2))
	print ("MSE for train: %.2f" % np.mean((clf.predict(trainMatrix) - traintarget) ** 2))
	f.close()

def pca(devMatrix, trainMatrix, devtarget, traintarget):

	print 'Running decomposition'
	svd = TruncatedSVD(n_components=1000)
	#trainMatrixTrans = svd.fit_transform(trainMatrix)
	#devMatrixTrans = svd.fit_transform(devMatrix)

	svd.fit(trainMatrix)
	trainMatrixTrans = svd.transform(trainMatrix)
	svd.fit(devMatrix)
	devMatrixTrans = svd.transform(devMatrix)
	print 'End Decomposition'
	#gradientBoost(devMatrixTrans, trainMatrixTrans, devtarget,traintarget)
	supportVectorMachine(devMatrixTrans,trainMatrixTrans,devtarget,traintarget)

def multinomialNB(devMatrix, trainMatrix, devtarget, traintarget):
	f = open('MNNB2.log', 'a')
	f.write("Making model!!!!!")
	print 'Making model!'
	clf = MultinomialNB(alpha=1, fit_prior=False)
	clf.fit(trainMatrix, traintarget)
	f.write("\n")
	value = ('Model: multinomial bayes with parameters ',clf.get_params(False))
	print (str(value))
	f.write(str(value))
	f.write("\n")
	f.write("MSE for train: %.2f" % np.mean((clf.predict(trainMatrix) - traintarget) ** 2))
	score = clf.score(trainMatrix, traintarget)
	f.write("\n")
	value = ('Score for train %.2f', score)
	f.write("\n")
	f.write("MSE for dev: %.2f" % np.mean((clf.predict(devMatrix) - devtarget) ** 2))
	score = clf.score(devMatrix, devtarget)
	value = ('Score for dev %.2f', score)
	print(str(value))
	f.write("\n")
	s = str(value)
	f.write(s)
	f.write("\n")
	f.write('model done')
	f.write("\n")
	f.write("\n")
	f.close()
	return score

def multiLinearReg(devMatrix, trainMatrix, devtarget, traintarget):
	f = open('MNLR2.log', 'a')
	#clf = linear_model.LogisticRegression(penalty='l2', dual=True, max_iter=200, solver='lbfgs', tol=0.001,multi_class='multinomial',verbose=1)
	#clf = linear_model.LogisticRegression( dual=True, max_iter=2000, solver='lbfgs', tol=0.001,multi_class='ovr',verbose=1)
	clf = linear_model.LogisticRegression(dual = True, max_iter=2500, solver='lbfgs', tol=0.001, multi_class='ovr', verbose=1, C=0.5)
	clf.fit(trainMatrix, traintarget)
	f.write("\n")
	value = ('Model: multinomial logistic regression with parameters ',clf.get_params(False))
	print (str(value))
	f.write(str(value))
	f.write("\n")
	f.write("MSE for train: %.2f" % np.mean((clf.predict(trainMatrix) - traintarget) ** 2))
	score = clf.score(trainMatrix, traintarget)
	f.write("\n")
	value = ('Score for train %.2f', score)
	f.write(str(value))
	f.write("\n")
	f.write("MSE for dev: %.2f" % np.mean((clf.predict(devMatrix) - devtarget) ** 2))
	score = clf.score(devMatrix, devtarget)
	value = ('Score for dev %.2f', score)
	print(str(value))
	f.write("\n")
	s = str(value)
	f.write(s)
	f.write("\n")
	f.write('model done')
	f.write("\n")
	f.write("\n")
	f.close()
	return score

def supportVectorMachine(devMatrix, trainMatrix, devtarget, traintarget):
	print ("hello")
	#linux 01 05 03 

	f = open ('svmlog64.txt', 'a')
	print 'Beginning model'
	f.write("beginning model \n")
	#parameters = [{ 'kernel':['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'], 'degree':[3,4,5,6], 
	#'gamma':[0.05 ,0.001 ,0.03 ,0.1 ,1 ,0.75 ,0.9 ,0.8 ,2.0, 5 ,0.25 , 1.3, 1.5, 1.75, 0.0001], 'coef0':[0.05 ,0.001 ,0.03 ,0.1 ,1 ,0.75 ,0.9 ,0.8 ,2.0, 5 ,0.25 , 1.3, 1.5, 1.75, 0.0001], 'max_iter':[-1]}]
	#clf = grid_search.GridSearchCV(svm.SVR(), parameters )
	clf = svm.SVR(kernel='rbf', cache_size=200, C=1.0, coef0=0.0, degree=3, epsilon=0.1, gamma=0.0, max_iter=-1, shrinking=True, tol=0.001, verbose=True)
	##clf = svm.SVR(kernel='rbf', cache_size=200, C=1.0, coef0=0.0, degree=3, epsilon=0.1, gamma=0.0, max_iter=-1, shrinking=True, tol=0.001, verbose=True, max_iter=-1 )
	#clf = svm.SVR(kernel='rbf', cache_size=1000, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001, max_iter=-1, shrinking=False, tol=0.0001, verbose=False )
	#clf = svm.SVR(kernel='rbf', cache_size=1000, coef0=0.0, degree=3, epsilon=0.01, gamma=0.001, max_iter=-1, shrinking=True, tol=0.0001, verbose=False )
	#clf = svm.SVR(kernel='sigmoid', cache_size=1000, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1, max_iter=-1, shrinking=True, tol=0.0001, verbose=False )
	#clf = svm.SVR(kernel='poly', cache_size=1000, coef0=0.1, degree=3, epsilon=0.1, gamma=0.1, max_iter=-1, shrinking=True, tol=0.0001, verbose=False )
	#clf = svm.SVR(kernel='poly', cache_size=1000, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1, max_iter=-1, shrinking=True, tol=0.0001, verbose=False )
	#clf = svm.NuSVR(kernel='rbf', cache_size=200, coef0=0.0, degree=3, nu=0.1, gamma=0.01, max_iter=-1, shrinking=True, tol=0.0001, verbose=False )
	#clf = svm.LinearSVR(loss='squared_epsilon_insensitive', dual=True, C=1.0, epsilon=0, max_iter=2000,  tol=0.0001, verbose=1 )
	f.write("model is made\n")

	clf.fit(trainMatrix, traintarget)
	print 'model finished'
	f.write("\n")
	value = ('Model: support vector machine with parameters ',clf.get_params(False))
	s = str(value)
	f.write(s)
	f.write("\n")

	f.write("MSE for train: %.2f" % np.mean((clf.predict(trainMatrix) - traintarget) ** 2))
	score = clf.score(trainMatrix, traintarget)
	f.write("\n")
	value = ('Score for train %.2f', score)
	f.write(str(value))
	f.write("\n")

	print(str(value))
	f.write("MSE for dev: %.2f" % np.mean((clf.predict(devMatrix) - devtarget) ** 2))
	score = clf.score(devMatrix, devtarget)
	value = ('Score for dev %.2f', score)
	print(str(value))
	f.write("\n")
	s = str(value)
	f.write(s)
	f.write("\n")

	f.write('model done')
	f.write("\n")

	return score


def plotSVM(devMatrix, trainMatrix, devtarget, traintarget):

	x = np.empty(trainMatrix.shape[0])
	for i in range(0, trainMatrix.shape[0]):
		x[i] = i
	# Fit regression model
	svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
	#svr_lin = svm.SVR(kernel='linear', C=1e3)
	#svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)
	y_rbf = svr_rbf.fit(trainMatrix, traintarget).predict(trainMatrix)
	#y_lin = svr_lin.fit(trainMatrix, traintarget).predict(trainMatrix)
	#y_poly = svr_poly.fit(trainMatrix, traintarget).predict(trainMatrix)

	###############################################################################
	# look at the results
	plt.scatter(x, traintarget, c='k', label='data')
	plt.hold('on')
	plt.plot(x ,y_rbf, c='g', label='RBF model')
	#plt.plot(x, y_lin, c='r', label='Linear model')
	#plt.plot(x, y_poly, c='b', label='Polynomial model')
	plt.xlabel('data')
	plt.ylabel('target')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

def elasticNet(devMatrix, trainMatrix, devtarget, traintarget):

	print("beginning grid search")
	f = open ('elasticNetlog.txt', 'a')
	f.write("beginning grid search \n")
	alphas = np.array([0.05 ,0.001 ,0.03 ,0.1 ,1 ,0.75 ,0.9 ,0.8 ,2.0, 5 ,0.25 , 1.3, 1.5, 1.75, 0.0001])
	parameters = [{'n_jobs':[-1], 'l1_ratio':[1, 0.75 ,0.5, 0.8, 0.9, 0.99, 0.95, 0.6], 'max_iter':[5, 10, 12, 15, 20, 200, 500, 1000] , 'normalize':[True, False]}]

	clf = grid_search.GridSearchCV(linear_model.ElasticNetCV(), parameters )

	f.write("grid search finished \n")

	clf.fit(trainMatrix, traintarget)
	f.write("\n")
	value = ('Model: elasticModel with parameters ',(clf.get_params(False)))
        s = str(value)
	f.write(s)
	f.write("\n")
	# f.write('Coefficients: ', clf.coef_)
	f.write("Residual sum of squares: %.2f"
	% np.mean((clf.predict(devMatrix) - devtarget) ))
	rsquared = clf.score(devMatrix, devtarget)
	value = ('R^2 value %.2f', rsquared)
	f.write("\n")
	s = str(value)
	f.write(s)
	f.write("\n")
	f.write('model done')
	f.write("\n")
	value1 = ('grid scores' ,(clf.grid_scores) , ' best estimator ', (best_estimator), ' best_score %.2f',(best_score) , "best params %.2f",(best_score) ," scorer ",(scorer))
	s = str(value1)
	f.write(s)
	f.write("\n")
	return rsquared

def ridgeReg(devMatrix, trainMatrix, devtarget, traintarget):
	parameters = [{'alpha':[0.05 ,0.001 ,0.03 ,0.1 ,1 ,0.75 ,0.9 ,0.8 ,2.0, 5 ,0.25 , 1.3, 1.5, 1.75, 0.0001] , 'max_iter':[5, 10, 12, 15, 20, 200, 500, 1000] , 'normalize':[True, False]}]
	print("Beginning grid search")
	f = open ('logRidge.txt', 'a')
	f.write("beginning grid search")
	clf = grid_search.GridSearchCV(linear_model.Ridge(), parameters )
	f.write("ending grid search")

	clf.fit(trainMatrix, traintarget)
	f.write("\n")
	value = ('Model: ridgeRegr with parameters ',(clf.get_params(False)))
	s = str(value)
        f.write(s)
	f.write("\n")
	# f.write('Coefficients: ', clf.coef_)
	f.write("Residual sum of squares: %.2f"
	% np.mean((clf.predict(devMatrix) - devtarget) ))
	rsquared = clf.score(devMatrix, devtarget)
	value = ('R^2 value %.2f', rsquared)
	f.write("\n")
	s = str(value)
	f.write(s)
	f.write("\n")
	f.write('model done')
	f.write("\n")
	value1 = ('grid scores' ,(clf.grid_scores) , ' best estimator ',(best_estimator), ' best_score %.2f',(best_score), "best params %.2f",(best_score) ," scorer ",(scorer))
	s = str(value1)
	f.write(s)
	f.write("\n")
	return rsquared

def lassoReg(devMatrix, trainMatrix, devtarget, traintarget):
	#parameters = [{'precompute':[True], 'alpha':[0.05 ,0.001 ,0.03 ,0.1 ,1 ,0.75 ,0.9 ,0.8 ,2.0, 5 ,0.25 , 1.3, 1.5, 1.75, 0.0001] , 'max_iter':[5, 10, 12, 15, 20, 200, 500, 1000] , 'normalize':[True, False], 'warm_start':[True, False]}]
	f = open ('logLasso4.txt', 'a')
	#f.write("Beginning grid search ")
	#f.write("\n")
	#print ("beginning grid search")
	#clf = grid_search.GridSearchCV(linear_model.Lasso(), parameters )
	#f.write("Ending grid search")
	#print("Ending grid search ")
	f.write("Beginning Fit ")
	print ("Beginning Fit ")

	#clf = linear_model.Lasso(alpha = 1.05, max_iter=1000, normalize=False, positive = False, precompute=False, random_state=None, tol=0.0001, warm_start=False, fit_intercept=True, copy_X=False, selection='cyclic')
	clf = linear_model.Lasso(alpha = 0.0001, max_iter=10000, normalize=False, positive = False, precompute=False, random_state=None, tol=0.0001, warm_start=False, fit_intercept=True, copy_X=False, selection='cyclic')
	clf.fit(trainMatrix, traintarget)
	f.write("\n")
	value = ('Model: lassoRegr with parameters ',(clf.get_params(False)))
	s = str(value)
	f.write(s)
	f.write("\n")
	# f.write('Coefficients: ', clf.coef_)ca 
	f.write("Residual sum of squares: %.2f" % np.mean((clf.predict(devMatrix) - devtarget) ** 2))
	rsquared = clf.score(devMatrix, devtarget)
	value = ('R^2 value %.2f', rsquared)
	f.write("\n")
	s = str(value)
	f.write(s)
	f.write("\n")

	# value1 = ('grid scores' ,(clf.grid_scores))
	# s = str(value1)
	# f.write(s)
	# f.write("\n")
	# value1 = (' best estimator ',(best_estimator))
	# s = str(value1)
	# f.write(s)
	# f.write("\n")
	# value1 = (' best_score %.2f',(best_score))
	# s = str(value1)
	# f.write(s)
	# f.write("\n")
	# value1 = ("best params %.2f",(best_score))
	# s = str(value1)
	# f.write(s)
	# f.write("\n")
	# value1 = (" scorer ",(scorer))
	# s = str(value1)
	# f.write(s)
	f.write('model done')
	f.write("\n")
	f.write("\n")
	f.close()
	return rsquared

def parseData():
	devtargetfile = open("proj_data/task4/dev.RT",'r')
	traintargetfile = open("proj_data/task4/train.RT",'r')
	smalltraindatafile = open("proj_data/task4/train.small.X",'r')
	traindatafile = open("proj_data/task4/train.sparseX",'r')
	devdatafile = open("proj_data/task4/dev.sparseX",'r')

	trainSmall = np.loadtxt(smalltraindatafile)
	traindata = np.loadtxt(traindatafile)
	devdata = np.loadtxt(devdatafile)
	traintarget = np.loadtxt(traintargetfile)
	traintargetfile.close()
	traindatafile.close()
	devtarget = np.loadtxt(devtargetfile)
	devtargetfile.close();
	print "Loading files done"


	trainSmallMatrix = trainSmall
	print "train small parsing done"


	row =[]
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
	print "train parsing done"


	row =[]
	col = []
	val = []

	for k in range(len(devdata)):
	    i = devdata[k][0]
	    j = devdata[k][1]
	    value = devdata[k][2]
	    row.append(int(i))
	    col.append(int(j))
	    val.append(int(value))
	print "parse done"
	devMatrix = coo_matrix((val, (row, col)), shape=(53379, 75000))

	devMatrix = csc_matrix(devMatrix)
	print "matrix done"
	return devMatrix, trainMatrix, devtarget, traintarget, trainSmallMatrix

def main():
	devMatrix, trainMatrix, devtarget, traintarget, trainSmallMatrix = parseData()
	#plotSVM(devMatrix, trainMatrix, devtarget, traintarget)
	#lossRidge = ridgeReg(devMatrix, trainMatrix, devtarget, traintarget)
	#lossElasticNet = elasticNet(devMatrix, trainMatrix, devtarget, traintarget)
	#svmLoss = supportVectorMachine(devMatrix, trainSmallMatrix, devtarget, traintarget)
	#print svmLoss
	#lossLasso = lassoReg(devMatrix, trainMatrix, devtarget, traintarget)
	#print lossRidge
	#print lossLasso
	#print lossElasticNet
	#lossMulti = multiLinearReg(devMatrix, trainMatrix, devtarget, traintarget)
	#print lossMulti
	#lossBayes = multinomialNB(devMatrix, trainMatrix, devtarget, traintarget)
	#print lossBayes
	#gradientBoost(devMatrix, trainMatrix, devtarget, traintarget)
	pca(devMatrix,trainMatrix,devtarget,traintarget) #which calls gradient boost

main()

# plt.scatter(devMatrix, devtarget,  color='black')
# plt.plot(devMatrix, regr.predict(devMatrix), color='blue',
#          linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()

#run serveral different models and tune them with different parameters
# output results to a file so that i can include them in report
# runs most sucessful models with regularization and pre-processing methods
