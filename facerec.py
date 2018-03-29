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
hogFaces = []

for face in range(grayfaces.shape[2]):
	hogFaces.append(hog(grayfaces[:,:,face], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)))

hogFaces = np.asarray(hogFaces)

forest = RandomForestClassifier(criterion='entropy',n_estimators=100,random_state=1,n_jobs=2)
forest.fit(hogFaces, range(grayfaces.shape[2]))

test_image = grayfaces[:, :, 2]
hog_test_image = hog(test_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

yhat = forest.predict(hog_test_image.reshape(1, -1))
print(yhat)
