import os
from glob import glob

from joblib import dump
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import cv2

from image_preprocessing import extract_hand

if __name__ == "__main__":
    features = []
    targets = []

    for number in range(6):
        files = glob(os.path.join('data', 'train', number, '*.png'))
        for image in files:
            features = extract_hand(image)

            #cv2.imshow('test', extracted_hand)
            #cv2.waitKey(0)

            features.append(features)
            targets.append(number)

    
    model = LinearSVC(dual=False)
    
    #x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

    #model.fit(x_train, y_train)
    #y_pred = model.predict(x_test)

    #print(accuracy_score(y_test, y_pred))
    #print(confusion_matrix(y_test, y_pred))

    model.fit(images, targets)
    dump(model, 'trained_model.joblib')
