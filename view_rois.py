import numpy as np
import PIL
import cv2

from PIL import ImageTk
from Tkinter import *
import random
import math
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

knn = cv2.KNearest()

rois = np.load('Teach_Data/SR1/A/10/coarse_learn_data.npy')
labels = np.load('Teach_Data/SR1/A/10/coarse_learn_labels.npy')

randomIndex = 12
imageData = rois[randomIndex]
num = labels[randomIndex]
im = PIL.Image.fromarray(imageData)

fd = hog(imageData, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=True)

