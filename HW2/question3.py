import cv2
import numpy as np

from HW2.question2 import question2
from HW2.question1 import question1


def question3(img_bw, img_color, blur_size=11):
    filtered = cv2.GaussianBlur(img_bw, (blur_size, blur_size), 0)
    edges = question1(img_bw)
    edges_f = question1(filtered)
    acc, img_lines = question2(filtered, img_color)
    return filtered, edges, edges_f, acc, img_lines


if __name__ == '__main__':
    im_color = cv2.imread("subject/ps1-input0-noise.png")
    im_bw = cv2.imread("subject/ps1-input0-noise.png", cv2.IMREAD_GRAYSCALE)
    flt, ed, edf, ac, il = question3(im_bw, im_color)
    cv2.imwrite("Images/ps1-3-a.png", flt)
    cv2.imwrite("Images/ps1-3-b-1.png", ed)
    cv2.imwrite("Images/ps1-3-b-2.png", edf)
    cv2.imwrite("Images/ps1-3-c-1.png", ac)
    cv2.imwrite("Images/ps1-3-c-2.png", il)

