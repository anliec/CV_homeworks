import cv2
import numpy as np
from scipy.signal import convolve2d
from math import floor


def int_img_to_unint8(img):
    max_value = np.max(np.abs(img))
    min_value = np.min(img)
    if min_value < 0.0:
        return np.array(img * 127 / max_value + 127, dtype=np.uint8)
    else:
        return np.array(img * 255 / max_value, dtype=np.uint8)


def compute_gradient(img: np.ndarray, direction: str):
    if direction == "X":
        imgA = np.array(img[:-1, :], dtype=np.int32)
        imgB = np.array(img[1:, :], dtype=np.int32)
    elif direction == "Y":
        imgA = np.array(img[:, :-1], dtype=np.int32)
        imgB = np.array(img[:, 1:], dtype=np.int32)
    else:
        raise ValueError("direction argument is expected to be 'X' or 'Y' only")

    return imgA - imgB


def harris_transform(img: np.ndarray, win_size: int, alpha=1.0):
    i_x = compute_gradient(img, 'X')
    i_y = compute_gradient(img, 'Y')

    # reshape both gradient so they have the same size
    xx, xy = i_x.shape
    yx, yy = i_y.shape
    i_x = i_x[:min(xx, yx), :min(xy, yy)]
    i_y = i_y[:min(xx, yx), :min(xy, yy)]

    win = np.ones(shape=(win_size, win_size))

    i_xx_sum = convolve2d(np.power(i_x, 2), win, 'valid')
    i_yy_sum = convolve2d(np.power(i_y, 2), win, 'valid')
    i_xy_sum = convolve2d(i_x * i_y,        win, 'valid')

    return i_xx_sum * i_yy_sum - i_xy_sum**2 + alpha * (i_xx_sum + i_yy_sum), i_x, i_y


def harris_filtering(harris_r: np.ndarray, number_of_points: int, win_size=5):
    r = harris_r.copy()
    r_w, r_h = r.shape
    r_filtered = np.zeros(shape=r.shape)
    win_radius = int(floor((win_size - 1) / 2))
    for i in range(number_of_points):
        p = np.argmax(r)
        p_x, p_y = p // r_h, p % r_h
        # r_filtered[p_x, p_y] = r[p_x, p_y]
        r_filtered[p_x, p_y] = 1
        r[max(0, p_x - win_radius):p_x + win_radius + 1, max(0, p_y - win_radius):p_y + win_radius + 1] = 0.0
    return r_filtered


if __name__ == '__main__':
    trans_a = cv2.imread("subject/transA.jpg", cv2.IMREAD_GRAYSCALE)
    trans_b = cv2.imread("subject/transB.jpg", cv2.IMREAD_GRAYSCALE)
    sim_a = cv2.imread("subject/simA.jpg", cv2.IMREAD_GRAYSCALE)
    sim_b = cv2.imread("subject/simB.jpg", cv2.IMREAD_GRAYSCALE)

    print("Question 1.1")

    trans_shape_x, trans_shape_y = trans_a.shape
    disp = np.zeros(shape=(trans_shape_x, trans_shape_y * 2))

    disp[1:, :trans_shape_y] = compute_gradient(trans_a, 'X')
    disp[:, trans_shape_y + 1:] = compute_gradient(trans_a, 'Y')

    cv2.imwrite("Images/ps4-1-1-a.png", int_img_to_unint8(disp))

    sim_shape_x, sim_shape_y = sim_a.shape
    disp = np.zeros(shape=(sim_shape_x, sim_shape_y * 2))

    disp[1:, :sim_shape_y] = compute_gradient(sim_a, 'X')
    disp[:, sim_shape_y + 1:] = compute_gradient(sim_a, 'Y')

    cv2.imwrite("Images/ps4-1-1-b.png", int_img_to_unint8(disp))

    print("Question 1.2 and 1.3")

    win_size = 10
    number_of_point = 200
    r, _, _ = harris_transform(trans_a, win_size)
    cv2.imwrite("Images/ps4-1-2-a.png", int_img_to_unint8(r))
    r_f = harris_filtering(r, number_of_point, win_size*2)
    cv2.imwrite("Images/ps4-1-3-a.png", int_img_to_unint8(r_f))

    r, _, _ = harris_transform(trans_b, win_size)
    cv2.imwrite("Images/ps4-1-2-b.png", int_img_to_unint8(r))
    r_f = harris_filtering(r, number_of_point, win_size*2)
    cv2.imwrite("Images/ps4-1-3-b.png", int_img_to_unint8(r_f))

    r, _, _ = harris_transform(sim_a, win_size)
    cv2.imwrite("Images/ps4-1-2-c.png", int_img_to_unint8(r))
    r_f = harris_filtering(r, number_of_point, win_size*2)
    cv2.imwrite("Images/ps4-1-3-c.png", int_img_to_unint8(r_f))

    r, _, _ = harris_transform(sim_b, win_size)
    cv2.imwrite("Images/ps4-1-2-d.png", int_img_to_unint8(r))
    r_f = harris_filtering(r, number_of_point, win_size*2)
    cv2.imwrite("Images/ps4-1-3-d.png", int_img_to_unint8(r_f))







