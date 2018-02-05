import cv2
import numpy as np

from HW2.question5 import *
from HW2.question2 import question2


if __name__ == '__main__':
    blur_size = 11
    im_bw = cv2.imread("subject/ps1-input3.jpg", cv2.IMREAD_GRAYSCALE)
    # part a
    filtered = cv2.GaussianBlur(im_bw, (blur_size, blur_size), 0)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(filtered, 50, 175, 3)
    acc = circular_hough_transform(edges)
    circle_image = find_and_draw_circle(acc.copy(), filtered.copy(), 0.65)
    cv2.imwrite("Images/ps1-8-a.png", circle_image)

    # part c
    height = 560
    width = 700
    pts_src = np.float32([[108, 32], [540, 20], [681, 278], [0, 285]])
    pts_dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    transform_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    inverse_transform_matrix = cv2.getPerspectiveTransform(pts_dst, pts_src)

    dst = cv2.warpPerspective(im_bw, transform_matrix, (width, height))

    filtered = cv2.GaussianBlur(dst, (blur_size, blur_size), 0)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(filtered, 10, 145, 3)
    acc = circular_hough_transform(edges, min_radius=60//2, max_radius=100//2)

    circle_image = find_and_draw_circle(acc.copy(), filtered.copy(), 0.5, min_radius=60//2)

    cv2.imwrite("Images/ps1-8-c-1.png", circle_image)

    original_height, original_width = im_bw.shape
    back_to_original_perspective = cv2.warpPerspective(circle_image, inverse_transform_matrix, (original_width, original_height))

    cv2.imwrite("Images/ps1-8-c-2.png", back_to_original_perspective)

    background = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2BGR)
    like_original = (back_to_original_perspective == 0) * background + back_to_original_perspective

    cv2.imwrite("Images/ps1-8-c-3.png", like_original)

