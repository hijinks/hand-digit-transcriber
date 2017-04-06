import fnmatch
import os
import numpy as np
import cv2
import idx2numpy

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

clf = LinearSVC()

knn = cv2.KNearest()

list_hog_fd = np.load('recent_hog_list.npy')
label_list = np.load('recent_label_list.npy')

list_hog_fd = np.asarray(np.concatenate(list_hog_fd).ravel().tolist())
list_hogs_np = []

for idx in range(len(list_hog_fd)):
    print(np.array(list_hog_fd[idx], dtype=float).shape)

print(np.array(list_hog_fd[1], dtype=float))
print(np.array(list_hog_fd[2], dtype=float))
clf.fit(list_hogs_np, label_list)

joblib.dump(clf, "digit_learn_all.pkl", compress=3)