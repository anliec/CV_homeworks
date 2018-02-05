import cv2
import numpy as np


def question1(img, th1=100, th2=100):
    edges = cv2.Canny(img, th1, th2)
    return edges


if __name__ == '__main__':
    im = cv2.imread("subject/ps1-input0.png")
    ed = question1(im)
    cv2.imwrite("Images/ps1-1-a.png", ed)
