import cv2
import numpy as np

def extract_hand(img, flatten=False, size=(50, 50)):
    # read image from stream
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # color segmentation
    min_YCrCb = np.array([45,140,50],np.uint8)
    max_YCrCb = np.array([255,170,120],np.uint8)
    img = cv2.inRange(img, min_YCrCb, max_YCrCb)

    # contour extraction and filling
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(img, contours, i, (255, 0, 0), thickness=cv2.FILLED)

    #img = cv2.bilateralFilter(img, 7, 50, 50)
    #img = cv2.GaussianBlur(img,(3, 3),0)
    img = cv2.resize(img, size)

    if flatten:
        return flatten_image(img)
    else:
        return img

def flatten_image(img):
    if type(img) == np.ndarray:
        pixels = img
    elif os.path.isfile(img):
        pixels = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    pixels //= pixels.max()
    return pixels.flatten()