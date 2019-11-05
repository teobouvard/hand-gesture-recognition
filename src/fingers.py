import os
from glob import glob

import cv2
import numpy as np

from joblib import load
from tqdm import tqdm
from image_preprocessing import extract_hand


if __name__ == "__main__":
    model = load('trained_model.joblib')
    video = cv2.VideoCapture('test.mp4')

    while(True):
        ret, frame = video.read()

        # extract hand features from current frame
        features = extract_hand(frame, flatten=True).reshape(1, -1)

        # predict the number of finger with pretrained model
        fingers = model.predict(features)[0]

        # display image and number of fingers
        frame = extract_hand(frame, size=(500, 500), crop=False)
        frame = cv2.putText(frame, fingers, (10, 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 255, 255))

        cv2.imshow('Fingers detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # release the capture
            video.release()
            cv2.destroyAllWindows()
            break
        

