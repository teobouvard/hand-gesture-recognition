import os
from datetime import datetime
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from image_preprocessing import extract_hand


def clean_data_dir():
    # remove all existing pictures in training directory
    path = os.path.join('data', 'train', '*', '*.jpg')
    images = glob(path)
    for image in images:
        os.remove(image)
    

def extract_data():
    videos = glob('videos/*')

    # for each video, extract hand at each frame and save the image in corresponding class folder
    for filename in sorted(videos):
        fingers = os.path.basename(filename)[0]
        video = cv2.VideoCapture(filename)

        while(video.isOpened()):
            ret, frame = video.read()
            if ret:

                hand = extract_hand(frame, crop=True, size=(50, 50))
                cv2.imshow(f'Extracting images - {fingers} fingers', hand)
                cv2.imwrite(os.path.join('data', 'train', fingers, f'{str(datetime.now())}.jpg'), hand)

                # handle keypress
                keypress = cv2.waitKey(1)
                if  keypress == ord('q'):
                    # stop extraction
                    exit()
                elif keypress == ord(' '):
                    # pause video 
                    cv2.waitKey(0)
                elif keypress == ord('s'):
                    # skip this video
                    break

            else:
                break
        
        # release the capture
        video.release()
        cv2.destroyAllWindows()    


if __name__ == "__main__":
    clean_data_dir()
    extract_data()
