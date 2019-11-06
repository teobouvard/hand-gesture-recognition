import os
from glob import glob

import cv2
import numpy as np

from tensorflow import keras
from tqdm import tqdm
from image_preprocessing import extract_hand

CONFIDENCE_THRESHOLD = 0.8


if __name__ == "__main__":
    model = keras.models.load_model('trained_classifier.h5')
    video = cv2.VideoCapture('test.mp4')

    while(True):
        ret, frame = video.read()

        # extract hand features from current frame
        features = extract_hand(frame, crop=True)
        if features.max() != 0:
            features /= 255.0
        features = np.expand_dims(features, 0)

        # predict the number of finger with pretrained model
        predictions = model.predict(features)[0]
        fingers = np.argmax(predictions)
        confidence = predictions[fingers]
        if confidence < CONFIDENCE_THRESHOLD :
            fingers = '?'

        # display image and number of fingers
        frame = extract_hand(frame, size=(500, 500), crop=False)
        frame = cv2.putText(frame, f'{str(fingers)} ({confidence:.2f})', (10, 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 255, 255))

        cv2.imshow('Fingers detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # release the capture
    video.release()
    cv2.destroyAllWindows()