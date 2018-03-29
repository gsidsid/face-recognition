"""QEA Module 2 Final Project

Facial Recognition using Random Forest Classification of Histograms of Oriented Gradients
Olin College 2017-2018

Siddharth Garimella and Utsav Gupta
"""

"""The RF + HOG process is widely regarded as one of the fastest and most accurate facial recognition
algorithms to date. RF Classification is particularly exceptional in object recognition, and the lightweight,
computationally inexpensive structure of decision trees allows for a quick classification process.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from skimage.feature import hog

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import pydot
from timeit import Timer

classdata = sio.loadmat('classdata.mat')

def train_test_split(X):
	data = classdata['y'][0,0]
	names = data[0][0]
	isSmile = data[1][0]
	grayfaces = classdata['grayfaces']
	train_x = grayfaces[:,:,0::2]
	test_x = grayfaces[:,:,1::2]
	return train_x, test_x

def getHOGFace(image):
	# represents each 256x256 image as 64 gradient vectors with 8 possible directions
	# L2-Hys Regularization performs superior to L1, suggesting features in the dataset are qualitatively different
	return hog(image, orientations=24, pixels_per_cell=(32, 32), cells_per_block=(1, 1), block_norm=('L2')) 

def validateAccuracy(classifier, test_x, verbose=True):
	HOG2 = []
	for face in range(test_x.shape[2]):
		HOG2.append(getHOGFace(test_x[:,:,face]))
	test_hogFaces = np.asarray(HOG2)
	print('Accuracy score: ' + str(classifier.score(test_hogFaces, range(test_x.shape[2])) * 100.00))
	"""
	misc = 0
	for i in range(test_x.shape[2]):
		test_hogFace = getHOGFace(test_x[:, :, i]).reshape(1,-1)
		yhat = classifier.predict(test_hogFace)
		if yhat != i:
			misc += 1
	accuracy_score = 100-(misc/test_x.shape[2])*100
	if verbose:
		print("Percent Accuracy: ", accuracy_score)
	return accuracy_score
	"""

def predictSuccess(classifier, selected_x, verbose=True):
	ind = classifier.predict(getHOGFace(test_x[:,:,selected_x]).reshape(1,-1))
	if verbose:
		print("Sample recognition success:", (ind == selected_x))

def exportTrees(forest, no_trees = 10):
	trees = forest.estimators_
	for idx, tree in enumerate(trees[:no_trees]):
		export_graphviz(tree, out_file='trees/estimator' + str(idx) + '.dot')
		(graph,) = pydot.graph_from_dot_file('trees/estimator' + str(idx) + '.dot')
		graph.write_png('trees/estimator' + str(idx) + '.png')


train_x, test_x = train_test_split(classdata)

HOG1 = []
for face in range(train_x.shape[2]):
	HOG1.append(getHOGFace(train_x[:,:,face]))
train_hogFaces = np.asarray(HOG1)



forest = RandomForestClassifier(criterion='entropy',n_estimators=600,random_state=123,n_jobs=-1,max_depth=8, max_leaf_nodes=2**8)
forest.fit(train_hogFaces, range(train_x.shape[2]))

predictSuccess(forest, 1)
t = Timer(lambda: predictSuccess(forest,1,verbose=False))

validateAccuracy(forest, test_x)
print('Executed in: ' + str(t.timeit(number=1)) + ' seconds.')

#exportTrees(forest,no_trees=15)


