import cv2

from HW2.question3 import question3
from HW2.question2 import hough_transform, draw_lines
from HW2.question1 import question1


def question6(im_bw, filter_size):
    filtered = cv2.GaussianBlur(src=im_bw,
                                ksize=(filter_size, filter_size),
                                sigmaX=0)
    edges = cv2.Canny(filtered, 40, 125)
    hough = hough_transform(edges)
    acc, img_line_a = draw_lines(hough, filtered, threshold=0.5)
    acc, img_line_c = draw_lines(hough, filtered, threshold=0.5, parallel_dist=(22, 35), parallel_threshold=0.7)
    return filtered, edges, img_line_a, img_line_c


if __name__ == '__main__':
    filter_size = 11
    im = cv2.imread("subject/ps1-input2.jpg", cv2.IMREAD_GRAYSCALE)
    flt, ed, ila, ilc = question6(im, filter_size)
    cv2.imwrite("Images/ps1-6-a.png", ila)
    cv2.imwrite("Images/ps1-6-c.png", ilc)
    cv2.imwrite("Images/ps1-6-a-test1.png", ed)

