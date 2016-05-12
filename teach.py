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
use_g_data = True

if os.path.isfile('recent_data.npy') and os.path.isfile('recent_label.npy'):
    data = np.load('recent_data.npy')
    labels = np.load('recent_labels.npy')
else:
    for root, dirnames, filenames in os.walk('/run/media/sam/3086-05D9/Teach_Data'):
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

    np.save('recent_data.npy', data)
    np.save('recent_labels.npy', data)

list_hog_fd = []
grouped_labels = []

print 'Process learning data'

if use_g_data:
    grouped_data = np.load('/run/media/sam/3086-05D9/Projects/hand-digit-transcriber/original_data.npy')
    grouped_labels = np.load('/run/media/sam/3086-05D9/Projects/hand-digit-transcriber/original_labels.npy')

    for feature in grouped_data:
         fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=True)
         list_hog_fd.append(fd)


for idx in range(len(data)):
    try:
        d = np.load(data[idx])
        l = np.load(labels[idx])
        if len(d) == len(l):
            for y in d:
                x = np.resize(y,(28,28))
                fd = hog(x.reshape(28, 28), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
                list_hog_fd.append(fd)
            grouped_labels = np.hstack([grouped_labels,l]).flatten()

    except:
        print 'Bad data file'
        print data[idx]



clf = LinearSVC()
label_list = grouped_labels.tolist()

print(len(label_list))
print(len(list_hog_fd))

np.save('recent_hog_list.npy', list_hog_fd)
np.save('recent_label_list.npy', label_list)

hog_features = np.array(list_hog_fd, dtype=float)
clf.fit(hog_features, label_list)

joblib.dump(clf, "digit_learn_all.pkl", compress=3)