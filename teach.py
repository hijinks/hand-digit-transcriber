import fnmatch
import os
import numpy as np
import cv2
import idx2numpy

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

knn = cv2.KNearest()

matches = {}
directories = []
for root, dirnames, filenames in os.walk('/Users/sambrooke/Documents/Academic/PhD/DV/Data'):
        #if filename in directories:
        #    directories.append(filename)
        #dk = directories.index(filename)


    for filename in fnmatch.filter(filenames, '*.npy'):

        if not root in directories:
            directories.append(root)

        k = directories.index(root)

        if not k in matches:
            matches.update({k: []})

        matches[k].append(filename)


data = []
labels = []

for k, v in matches.iteritems():
    t = []

    for n in v:
        l = n.split('_learn_')
        t.append(l[0])

    e = np.unique(t)
    d = directories[k]
    for u in t:
        data.append(d+'/'+u+'_learn_data.npy')
        labels.append(d+'/'+u+'_learn_labels.npy')


# Load original the dataset
grouped_data = np.load('/Users/sambrooke/Repos/digit_recognition/original_data.npy')
grouped_labels = np.load('/Users/sambrooke/Repos/digit_recognition/original_labels.npy')

list_hog_fd = []

for feature in grouped_data:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)

for df in data:
    d = np.load(df)
    for y in d:
        x = np.resize(y,(28,28))
        fd = hog(x.reshape(28, 28), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(fd)

for lf in labels:
     l = np.load(lf)
     grouped_labels = np.hstack([grouped_labels,l]).flatten()


clf = LinearSVC()
label_list = grouped_labels.tolist()
print(len(label_list))
print(len(list_hog_fd))
hog_features = np.array(list_hog_fd, dtype=float)
clf.fit(hog_features, label_list)

joblib.dump(clf, "digit_learn.pkl", compress=3)