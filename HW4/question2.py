import numpy as np
import cv2

from HW4.utils import bmatrix


def gen_a(zipped_point):
    n = len(zipped_point)

    a = np.zeros(shape=(n, 9), dtype=np.double)

    for i, ((a_x, a_y), (b_x, b_y)) in enumerate(zipped_point):
        a[i, :] = [a_x * b_x, a_x * b_y, a_x,
                   a_y * b_x, a_y * b_y, a_y,
                   b_x,       b_y,       1]
    return a


def compute_projection_matrix(zipped_point):
    a = gen_a(zipped_point)
    ata = np.matmul(a.T, a)
    val, vec = np.linalg.eig(ata)
    i = np.argmin(val)
    sol = vec[:, i]
    factor = 1 / np.sum(np.abs(np.power(sol, 2)))
    sol *= factor
    return np.reshape(sol, newshape=(3, 3))


def reduce_rank(f):
    u, d, v = np.linalg.svd(f)
    d[2] = 0.0
    return np.mat(u) * np.mat(np.diag(d)) * np.mat(v)


def point_of_line(p, f):
    ph = np.mat([0, 0, 1])
    ph[0, :2] = p[:2]
    fx = np.mat(f) * ph.T
    b = 0
    p1 = ((b * fx[1] + fx[2]) / -fx[0], b)
    b = 712
    p2 = ((b * fx[1] + fx[2]) / -fx[0], b)
    return p1, p2


def draw_lines(img, src_pts, f):
    for pt in src_pts:
        p1, p2 = point_of_line(pt, f)
        p1 = tuple(map(int, p1))
        p2 = tuple(map(int, p2))
        cv2.line(img, p1, p2, (255, 0, 0), 1)
    return img


if __name__ == '__main__':
    print("question 2.1")
    points_pic_a = []
    with open("subject/pts2d-pic_a.txt", 'r') as file:
        for l in file:
            point = l.split()
            points_pic_a.append(tuple(map(float, point)))
    points_pic_b = []
    with open("subject/pts2d-pic_b.txt", 'r') as file:
        for l in file:
            point = l.split()
            points_pic_b.append(tuple(map(float, point)))
    full_list = list(zip(points_pic_b, points_pic_a))

    f = compute_projection_matrix(full_list)

    print(bmatrix(f))

    fc = reduce_rank(f)
    print(bmatrix(fc))

    im_a = cv2.imread("subject/pic_a.jpg")
    im_a = draw_lines(im_a, points_pic_b, fc.T)
    cv2.imwrite("Images/ps3-2-3-a.png", im_a)

    im_b = cv2.imread("subject/pic_b.jpg")
    im_b = draw_lines(im_b, points_pic_a, fc)
    cv2.imwrite("Images/ps3-2-3-b.png", im_b)



