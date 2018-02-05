import cv2
import numpy as np

from HW2.question1 import question1

HOUGH_THRESHOLD = 0.5


def question2(img, full_image=None):
    edges = question1(img)
    acc = hough_transform(edges)
    if full_image is not None:
        return draw_lines(acc, full_image)
    else:
        return draw_lines(acc, img)


def draw_lines(acc, img, threshold=0.5, parallel_dist=None, parallel_threshold=0.7):
    acc = acc.copy()
    img = img.copy()
    max_vote = np.max(acc)
    max_points_d, max_points_theta = (acc > threshold * max_vote).nonzero()
    diag, _ = acc.shape
    for theta, d in zip(max_points_theta, max_points_d):
        if parallel_dist is not None:
            d_min_up = d + parallel_dist[0]
            d_max_up = d + parallel_dist[1]
            d_min_down = d - parallel_dist[1]
            d_max_down = d - parallel_dist[0]
            if np.max(acc[d_min_up:d_max_up, theta]) < parallel_threshold * acc[d, theta] \
                    and (d_min_down < 0 or np.max(acc[d_min_down:d_max_down, theta]) < parallel_threshold * acc[d, theta]):
                continue
        a = 2.0 * np.pi * theta / diag
        theta_pi_over_two = int(np.floor(diag / 4.0))
        theta_pi = int(np.floor(diag / 2.0))
        # compute position of two point on the line
        if theta % theta_pi == theta_pi_over_two:
            p1 = np.array([diag, d])
        else:
            p1 = np.array([int(d / np.cos(a)), 0])
        if theta % theta_pi == 0:
            p2 = np.array([d, diag])
        else:
            p2 = np.array([0, int(d / np.sin(a))])
        # ensure that the line between the two points is in the screen
        if p1[0] < 0:
            p1 += (p2 - p1) * 10
        if p2[1] < 0:
            p2 += (p1 - p2) * 10
        cv2.line(img, tuple(p1), tuple(p2), (255, 0, 0), 2)
        cv2.circle(acc, (theta, d), 10, (max_vote, 0, 0), 1)
    acc *= (255/max_vote)
    return acc, img


def hough_transform(edges):
    edges_points_y, edges_points_x = edges.nonzero()
    diag = int(np.floor(np.linalg.norm(edges.shape)))
    accumulator = np.zeros(shape=(diag, diag))
    for x, y in zip(edges_points_x, edges_points_y):
        # theta = np.mgrid[:diag]
        # a = np.pi * theta / diag
        # sin_line = x * np.cos(a) + y * np.sin(a)
        # sin_line = np.floor(sin_line)
        # accumulator += (sin_line == d)
        for theta in range(0, diag):
            a = 2.0 * np.pi * theta / diag
            d = x * np.cos(a) + y * np.sin(a)
            d = int(np.floor(d))
            if 0 <= d < diag:
                accumulator[d][theta] += 1
    return accumulator


if __name__ == '__main__':
    im = cv2.imread("subject/ps1-input0.png")
    acc, im = question2(im)
    cv2.imwrite("Images/ps1-2-b.png", im)
    cv2.imwrite("Images/ps1-2-a.png", acc)
