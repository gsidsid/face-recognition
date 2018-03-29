"""QEA Module 2 Final Project

Facial Recognition using Random Forest Classification of Histograms of Oriented Gradients
Olin College 2017-2018

Siddharth Garimella and Utsav Gupta
"""

"""The RF + HOG Ensemble is widely regarded as one of the fastest and most accurate facial recognition
algorithms to date. RF Classification is particularly exceptional in object recognition, and the lightweight,
computationally inexpensive structure of decision trees allows for a quick classification process.
"""

from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

classdata = sio.loadmat('classdata.mat')

data = classdata['y'][0,0]
names = data[0][0]
isSmile = data[1][0]

grayfaces = classdata['grayfaces']

train_x = grayfaces[:,:,0::2]
test_x = grayfaces[:,:,1::2]

print(train_x.shape)
print(test_x.shape)

train_hogFaces = []

for face in range(train_x.shape[2]):
	train_hogFaces.append(hog(train_x[:,:,face], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)))

train_hogFaces = np.asarray(train_hogFaces)

forest = RandomForestClassifier(criterion='entropy',n_estimators=100,random_state=1,n_jobs=2)
forest.fit(train_hogFaces, range(test_x.shape[2]))

test_image = 8
test_hogFace = hog(test_x[:, :, test_image], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

yhat = forest.predict(test_hogFace.reshape(1, -1))
print(yhat)
