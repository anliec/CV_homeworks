import numpy as np
import random

from HW4.utils import bmatrix


def gen_a(zipped_point):
    n = len(zipped_point)

    a = np.zeros(shape=(2 * n, 12), dtype=np.double)

    for i, ((p_x, p_y), (w_x, w_y, w_z)) in enumerate(zipped_point):
        a[2 * i, :] = [w_x, w_y, w_z, 1, 0, 0, 0, 0, -p_x * w_x, -p_x * w_y, -p_x * w_z, -p_x]
        a[2 * i + 1, :] = [0, 0, 0, 0, w_x, w_y, w_z, 1, -p_y * w_x, -p_y * w_y, -p_y * w_z, -p_y]

    return a


def compute_projection_matrix(zipped_point):
    a = gen_a(zipped_point)
    ata = np.matmul(a.T, a)
    val, vec = np.linalg.eig(ata)
    i = np.argmin(val)
    sol = vec[:, i]
    factor = 1 / np.sum(np.abs(np.power(sol, 2)))
    sol *= factor
    return np.reshape(sol, newshape=(3, 4))


def compute_m_from_sample(full_point_list):
    min_diff = None
    best = None
    diff_list = []
    for s in [8, 12, 16]:
        for i in range(10):
            random.shuffle(full_point_list)
            m = compute_projection_matrix(full_point_list[:s])
            diff = 0.0
            for n in range(1, 5):
                p = full_point_list[-n]
                p3d = np.ones(shape=4)
                p3d[0:3] = p[1]
                p2d = np.ones(shape=3)
                p2d[0:2] = p[0]
                conv = np.matmul(m, p3d)
                conv /= conv[2]
                diff += np.sum(np.power(p2d - conv, 2))
            if min_diff is None or min_diff > diff:
                min_diff = diff
                best = (m, s, i)
            diff_list.append(diff / 4)
    return best, diff_list


if __name__ == '__main__':
    print("question 1.1")
    points_3d = []
    with open("subject/pts3d-norm.txt", 'r') as file_3d:
        for l in file_3d:
            point = l.split()
            points_3d.append(tuple(map(float, point)))
    points_2d = []
    with open("subject/pts2d-norm-pic_a.txt", 'r') as file_3d:
        for l in file_3d:
            point = l.split()
            points_2d.append(tuple(map(float, point)))
    full_list = list(zip(points_2d, points_3d))

    m1 = compute_projection_matrix(full_list)
    print(bmatrix(m1))

    last_3d_point = np.ones(shape=4)
    last_3d_point[:3] = points_3d[-1]
    p = np.matmul(m1, last_3d_point)
    p /= p[2]
    print("(u,v):\n", bmatrix(p[:2]))

    print("question 1.2")
    points_3d = []
    with open("subject/pts3d.txt", 'r') as file_3d:
        for l in file_3d:
            point = l.split()
            points_3d.append(tuple(map(float, point)))
    points_2d = []
    with open("subject/pts2d-pic_b.txt", 'r') as file_3d:
        for l in file_3d:
            point = l.split()
            points_2d.append(tuple(map(float, point)))
    full_list = list(zip(points_2d, points_3d))

    (m, s, i), residual_list = compute_m_from_sample(full_list)

    print("best m:")
    print(bmatrix(m))
    print(s, i)
    residual_matrix = np.array(residual_list).reshape(3, 10).T
    print("list of residual:\n", bmatrix(residual_matrix))

    print("question 1.3")

    c = - np.matmul(np.linalg.inv(m[0:3, 0:3]), m[:, 3])

    print("c:\n", bmatrix(c))

