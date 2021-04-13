import os
import json
import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # load image
    img_path = "test.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    #cv2.imshow("img", thresh)

    _, labels, status, centroids = cv2.connectedComponentsWithStats(thresh)

    ltImage = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    rtImage = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    if status[1][0] < status[2][0]:
        roi1 = thresh[status[1][1]:status[1][1] + status[1][3], status[1][0]:status[1][0] + status[1][2]]
        roi2 = thresh[status[2][1]:status[2][1] + status[2][3], status[2][0]:status[2][0] + status[2][2]]
    else:
        roi2 = thresh[status[1][1]:status[1][1] + status[1][3], status[1][0]:status[1][0] + status[1][2]]
        roi1 = thresh[status[2][1]:status[2][1] + status[2][3], status[2][0]:status[2][0] + status[2][2]]
    #cv2.imshow("roi1",roi1)
    #cv2.imshow("roi2",roi2)

    # calcu blob
    _,contours, hierarchy = cv2.findContours(roi1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area1 = cv2.contourArea(contours[0])
    length1 = cv2.arcLength(contours[0], True)
    contours, hierarchy = cv2.findContours(roi2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area2 = cv2.contourArea(contours[0])
    length2 = cv2.arcLength(contours[0], True)

    #area
    imgArea = (area1 + area2) / 2
    realArea = imgArea * 49 / 40000
    #height
    imgHeight = (status[1][3] + status[2][3]) / 2
    realHeight = imgHeight * 7 / 20
    #length
    imgLength = (length1 + length2) / 2
    realLength = imgLength * 7 / 20
    #roundness
    r1 = (4 * math.pi * area1) / (length1 * length1)
    r2 = (4 * math.pi * area2) / (length2 * length2)
    roungness = (r1 + r2) / 2

    print("the area is %f cm*cm" % realArea)
    print("the height is %f mm" % realHeight)
    print("the length is %f mm" % realLength)
    print("the roundness is %f" % roungness)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
