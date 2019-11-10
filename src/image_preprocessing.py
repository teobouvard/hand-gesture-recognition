import cv2
import numpy as np
import os

MIN_HAND_SIZE = 100000

def extract_hand(img, crop, size=(50, 50)):
    # convert to HSV
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

    # contour extraction and bounding box
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > MIN_HAND_SIZE:
            img = bound_box(img, c, crop)

    # resizing
    img = cv2.resize(img, size)

    # filtering
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)

    return img


def bound_box(img, contour, crop):
    # get hand bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # center box 
    if h > w:
        min_x = x + (w - h) // 2
        max_x = x + (w + h) // 2
        min_y = y
        max_y = y + h
    else:
        min_x = x
        max_x = x + w
        min_y = y + (h - w) // 2
        max_y = y + (h + w) // 2

    if crop:
        return img[min_y:max_y, min_x:max_x]
    else:
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255,0,0), 3)
        return img