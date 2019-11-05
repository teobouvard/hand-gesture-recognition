import cv2
import numpy as np
import os

MIN_HAND_SIZE = 10000

def extract_hand(img, flatten=True, size=(500, 500), crop=True):
    # read image from stream
    if type(img) == np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif os.path.isfile(img):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create kernel for morphological operations
    kernel = np.ones((3,3), np.uint8)
    
    # extract hand
    min_range = np.array([0,100,50],np.uint8)
    max_range = np.array([50,255,255],np.uint8)
    img = cv2.inRange(img, min_range, max_range)

    # opening to remove noise 
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)

    # closing to fill gaps in hand
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)

    # contour extraction and filling
    if crop:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > MIN_HAND_SIZE:
                img = crop_image(img)

    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = cv2.resize(img, size)

    if flatten:
        img = flatten_image(img)

    return img


def flatten_image(img):
    if img.max() != 0:
        img /= img.max()
    return img


def crop_image(img):
    mask = img > 0
    return img[np.ix_(mask.any(1), mask.any(0))]