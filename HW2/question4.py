import cv2
import numpy as np

from HW2.question3 import question3


if __name__ == '__main__':
    im_bw = cv2.imread("subject/ps1-input1.jpg", cv2.IMREAD_GRAYSCALE)
    im_draw = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2BGR)
    flt, _, edf, ac, il = question3(im_bw, im_draw, 11)
    cv2.imwrite("Images/ps1-4-a.png", flt)
    cv2.imwrite("Images/ps1-4-b.png", edf)
    cv2.imwrite("Images/ps1-4-c-1.png", ac)
    cv2.imwrite("Images/ps1-4-c-2.png", il)

