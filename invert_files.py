import cv2
import glob
import os
import numpy as np

image_dir = r"/media/sb708/Maxtor/Fan Data/2016_data/fans/GC"

images = glob.glob(image_dir+'/*.jpg')

for i in images:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    cv2.imwrite(i, img)

