import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
#
#
# c1r = np.hstack(c1r)
# c2r = np.hstack(c2r)
#
#
# c3 = np.setdiff1d(c2r, c1r)
# df = pd.read_csv('/run/media/sam/SAM_DATA/counts/G8/C/4/fines_wolman.csv')
# c1 = df.ix[:,0:1].values.tolist()
# c2 = df.ix[:,1:2].values.tolist()
#
# c1r = np.array(c1)
# c2r = np.array(c2)
# print(c3)

hog_list = np.load('recent_hog_list.npy').flatten()
label_list = np.load('recent_label_list.npy')

print(type(label_list))
print(type(hog_list))
clf = LinearSVC()

hog_features = np.array(hog_list, dtype=float)
clf.fit(hog_features, label_list)