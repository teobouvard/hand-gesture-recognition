import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm

SIZE = (50, 50)

def extract_hand(path):
   # read image as grayscale and resize it
    img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, SIZE)

    # blur
    img = cv2.GaussianBlur(img,(5,5),0)

    # OTSU threshold
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # binarize data
    img = img // img.max()

    return img.flatten()



if __name__ == "__main__":
    images = []
    targets = []

    for number in tqdm(os.listdir('data')):
        files = glob(os.path.join('data', number, '*.png'))
        images.extend(map(extract_hand, files))
        targets.extend([number] * len(files))

    x_train, x_test, y_train, y_test = train_test_split(images, targets, test_size = 0.2, random_state = 42)

    model = LinearSVC(dual=False)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # show image
    #cv2.imshow('hand', hand)
    #cv2.waitKey(0)
