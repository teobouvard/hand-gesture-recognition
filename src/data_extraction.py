import os

import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from image_preprocessing import extract_hand

if __name__ == "__main__":
    video = cv2.VideoCapture('test.mp4')

    while(True):
        ret, frame = video.read()

        hand = extract_hand(frame, crop=True, flatten=False)
        cv2.imshow('image', hand)
        #cv2.imwrite(os.path.join('video_snapshots', f'{str(datetime.now())}.jpg'), hand)
        

        keypress = cv2.waitKey(1)
        if  keypress == ord('q'):
            # release the capture
            video.release()
            cv2.destroyAllWindows()
            break
        elif keypress == ord(' '):
            cv2.waitKey(0)
