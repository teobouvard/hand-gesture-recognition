from datetime import datetime
import os
from glob import glob

import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from image_preprocessing import extract_hand

CONFIDENCE_THRESHOLD = 0.9


if __name__ == "__main__":
    # load pre-trained model and open test video
    model = keras.models.load_model('trained_classifier.h5')
    video = cv2.VideoCapture('data/test.mp4')

    while(video.isOpened()):
        ret, frame = video.read()
        if ret:

            # extract hand features from current frame
            features = extract_hand(frame, crop=True)

            # create batch with single instance
            features = np.expand_dims(features, 0)

            # predict the number of finger with pretrained model
            predictions = model.predict(features)[0]
            fingers = np.argmax(predictions)
            confidence = predictions[fingers]

            # do not show prediction if confidence is too low
            if confidence < CONFIDENCE_THRESHOLD :
                fingers = '?'

            # display image and number of fingers
            frame = extract_hand(frame, size=(1000, 600), crop=False)
            frame = cv2.putText(frame, f'{str(fingers)} ({confidence:.2f})', (10, 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 255, 255))
            cv2.imshow('Fingers detection', frame)

            # handle keypress
            keypress = cv2.waitKey(1)
            if  keypress == ord('q'):
                break     
            elif keypress == ord('s'):
                # snapshot
                cv2.imwrite(os.path.join('demo_img', f'{str(datetime.now())}.jpg'), frame)

        else:
            # loop video
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # release the capture on exit
    video.release()
    cv2.destroyAllWindows()
