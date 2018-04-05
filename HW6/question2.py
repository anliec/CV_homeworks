import numpy as np
import cv2
from scipy.signal import convolve2d
import os
import matplotlib.pyplot as plt

from HW5.question1 import compute_gradient, int_img_to_unint8
from HW6.question1 import gaussian_reduce


def load_folder(path: str, blur=None):
    seq = list()
    paths = list()
    fs_path = os.fsencode(path)
    for file in os.listdir(fs_path):
        file_name = os.fsdecode(file)
        paths.append(os.path.join(path, file_name))
    paths.sort()
    for file in paths:
        im = cv2.imread(filename=file,
                        flags=cv2.IMREAD_GRAYSCALE)
        if blur is not None:
            im = cv2.GaussianBlur(im, (blur, blur), 1, borderType=cv2.BORDER_REPLICATE)
        seq.append(im)
    return seq, paths


def lucas_kanade_optic_flow(im1, im2, win_size, blur_size=None):
    if blur_size is not None:
        im1 = cv2.GaussianBlur(im1, (blur_size, blur_size), 1, borderType=cv2.BORDER_REPLICATE)
        im2 = cv2.GaussianBlur(im2, (blur_size, blur_size), 1, borderType=cv2.BORDER_REPLICATE)

    i_x = compute_gradient(im1, 'X')
    i_y = compute_gradient(im1, 'Y')
    i_t = im1.astype(np.int32) - im2.astype(np.int32)

    win = cv2.getGaussianKernel(win_size, 1)

    i_xx_sum = convolve2d(np.power(i_x, 2), win, 'same', boundary='symm')
    i_yy_sum = convolve2d(np.power(i_y, 2), win, 'same', boundary='symm')
    i_xy_sum = convolve2d(i_x * i_y, win, 'same', boundary='symm')
    i_xt_sum = convolve2d(i_x * i_t, win, 'same', boundary='symm')
    i_yt_sum = convolve2d(i_y * i_t, win, 'same', boundary='symm')

    det = np.power(i_xx_sum * i_yy_sum - i_xy_sum**2, -1)

    inds = np.where(np.isinf(det))
    det[inds] = 0.0

    vx = (-i_yy_sum * i_xt_sum + i_xy_sum * i_yt_sum) * det
    vy = ( i_xy_sum * i_xt_sum - i_xx_sum * i_yt_sum) * det

    return vx, vy


def wrap(im: np.ndarray, vx: np.ndarray, vy: np.ndarray):
    map1 = vx + np.array(range(vx.shape[0])).reshape((vx.shape[0], 1))
    map2 = vy + np.array(range(vy.shape[1])).reshape((1, vy.shape[1]))
    im = im[:vx.shape[0], :vx.shape[1]]
    ret = cv2.remap(im, map2.astype(np.float32), map1.astype(np.float32), cv2.INTER_NEAREST)
    return ret


def quiver(im: np.ndarray, u: np.ndarray, v: np.ndarray, path: str, scale=0.1):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2RGB),
              aspect='auto',
              extent=(0, im.shape[1] * scale, 0, im.shape[0] * scale))
    u = cv2.resize(u, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    v = cv2.resize(v, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ax.quiver(np.flipud(v), -np.flipud(u), color='c')
    plt.savefig(path)


if __name__ == '__main__':
    shift0 = cv2.imread("subject/TestSeq/Shift0.png", cv2.IMREAD_GRAYSCALE)
    shift0 = cv2.GaussianBlur(shift0, (51, 51), 1)

    # 2.1 and 2.2

    for e, i in enumerate(['2', '10', '20', '40', '5U5']):
        shift_i = cv2.imread("subject/TestSeq/ShiftR"+i+".png", cv2.IMREAD_GRAYSCALE)
        shift_i = cv2.GaussianBlur(shift_i, (51, 51), 1)
        u, v = lucas_kanade_optic_flow(shift0, shift_i, 51)
        # im = wrap(shift0, u, v)
        quiver(shift0, u, v, "Images/ps5-2-1-R" + i + ".png")
        # cv2.imwrite("Images/zdebug0_R" + i + "w.png", im)

    # 2.3

    seq1, paths1 = load_folder("subject/DataSeq1")
    seq2, paths2 = load_folder("subject/DataSeq2")

    for l in range(1):
        seq1 = list(map(gaussian_reduce, seq1))

    for i, (p, n) in enumerate(zip(seq1[:-1], seq1[1:])):
        u, v = lucas_kanade_optic_flow(p, n, 51)
        im = wrap(p, u, v)
        cv2.imwrite("Images/ps5-2-3-" + str(i) + ".png", int_img_to_unint8(im.astype(np.int) - n.astype(np.int)))
        quiver(im, u, v, "Images/ps5-2-3-" + str(i) + "q.png", 0.3)

    for l in range(2):
        seq2 = list(map(gaussian_reduce, seq2))

    for i, (p, n) in enumerate(zip(seq2[:-1], seq2[1:])):
        u, v = lucas_kanade_optic_flow(p, n, 51)
        im = wrap(p, u, v)
        cv2.imwrite("Images/ps5-2-3-" + str(i + len(seq1) - 1) + ".png", int_img_to_unint8(im.astype(np.int) - n.astype(np.int)))
        quiver(im, u, v, "Images/ps5-2-3-" + str(i + len(seq1) - 1) + "q.png", 0.3)

