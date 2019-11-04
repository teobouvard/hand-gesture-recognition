import os
from glob import glob

import cv2
import numpy as np
from sklearn.svm import LinearSVC
from tqdm import tqdm

SIZE = (50, 50)

def extract_hand(img, flatten=True):
    # read image as grayscale and resize it
    if type(img) == str:
        img = cv2.imread(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, SIZE)

    # blur
    img = cv2.GaussianBlur(img,(5,5),0)

    # OTSU threshold
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if flatten:
        # binarize data
        img = img // img.max()
        return img.flatten()
    else:
        return img



if __name__ == "__main__":
    #images = []
    #targets = []
#
    #for number in tqdm(os.listdir('data')):
    #    files = glob(os.path.join('data', number, '*.png'))
    #    images.extend(map(extract_hand, files))
    #    targets.extend([number] * len(files))
#
    #model = LinearSVC(dual=False)
    #model.fit(images, targets)

    video = cv2.VideoCapture(0)
    back_sub = cv2.createBackgroundSubtractorKNN()

    while(True):
        ret, frame = video.read()

        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            #frame = back_sub.apply(frame, 1)
            #x = model.predict(extract_hand(frame).reshape(1, -1))
            #print(x)

            #hand = extract_hand(frame, flatten=False)
            cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()