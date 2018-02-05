import cv2
import numpy as np

from HW2.question1 import question1

MIN_RADIUS = 35 // 2
MAX_RADIUS = 80 // 2


def question5(img_bw, img_color, blur_size=11, canny_th1=100, canny_th2=100, circle_th=0.5, min_radius=MIN_RADIUS, max_radius=MAX_RADIUS):
    img = img_bw.copy()
    filtered = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    edges = question1(filtered, canny_th1, canny_th2)
    acc = circular_hough_transform(edges, min_radius, max_radius)
    circle_image = find_and_draw_circle(acc, img_color.copy(), circle_th,
                                        min_radius=min_radius)
    return edges, circle_image


def circular_hough_transform(edges, min_radius=MIN_RADIUS, max_radius=MAX_RADIUS):
    h, w = edges.shape
    acc = np.zeros(shape=(max_radius - min_radius, h, w))
    edges_points_x, edges_points_y = edges.nonzero()
    zeros = np.zeros
    draw_circle = cv2.circle
    for x, y in zip(edges_points_x, edges_points_y):
        assert x < h and y < w
        for r in range(min_radius, max_radius):
            plane = zeros((h, w))
            draw_circle(plane, (y, x), r, (1, 0, 0), 1)
            acc[r - min_radius] += plane
    return acc


def find_and_draw_circle(acc, img, display_limit=0.5, min_radius=MIN_RADIUS):
    d, _, _ = acc.shape
    for r in range(0, d):
        acc[r] /= (r + min_radius)
    max_vote = np.max(acc)
    max_r, max_x, max_y = (acc >= max_vote * display_limit).nonzero()
    for r, x, y in zip(max_r, max_x, max_y):
        cv2.circle(img, (y, x), r + min_radius, (255, 0, 0), 2)
    return img


if __name__ == '__main__':
    im = cv2.imread("subject/ps1-input1.jpg")
    im_bw = cv2.imread("subject/ps1-input1.jpg", cv2.IMREAD_GRAYSCALE)
    ed, cr = question5(im_bw, im,
                       min_radius=34//2,
                       max_radius=64//2)
    cv2.imwrite("Images/ps1-5-a-1.png", ed)
    cv2.imwrite("Images/ps1-5-a-2.png", cr)
