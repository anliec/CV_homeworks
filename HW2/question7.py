import cv2
import numpy as np

from HW2.question5 import question5


if __name__ == '__main__':
    blur_size = 11
    im_bw = cv2.imread("subject/ps1-input2.jpg", cv2.IMREAD_GRAYSCALE)
    filtered = cv2.GaussianBlur(im_bw, (blur_size, blur_size), 0)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    ed, cr = question5(im_bw, filtered, blur_size, 30, 185, 0.55,
                       min_radius=50//2,
                       max_radius=80//2
                       )
    cv2.imwrite("Images/ps1-7-a.png", cr)
