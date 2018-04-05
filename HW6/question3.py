import numpy as np
import cv2
from scipy.signal import convolve2d
import os

from HW6.question2 import lucas_kanade_optic_flow, wrap, load_folder, quiver
from HW6.question1 import gaussian_reduce, gaussian_expend
from HW5.question1 import int_img_to_unint8


def hierarchical_lk(im1, im2, level, win_size=5, blur_size=None):
    sr = []
    imgs = (im1, im2)
    sr.append(imgs)
    for lvl in range(level):
        imgs = gaussian_reduce(imgs[0]), gaussian_reduce(imgs[1])
        sr.append(imgs)

    vx = np.zeros(sr[-1][0].shape, dtype=np.float64)
    vy = np.zeros(sr[-1][0].shape, dtype=np.float64)
    for lvl in range(level, -1, -1):
        w, h = sr[lvl][0].shape
        vx = vx[:w, :h]
        vy = vy[:w, :h]
        im_l0 = wrap(sr[lvl][0], vx, vy)
        new_vx, new_vy = lucas_kanade_optic_flow(im_l0, sr[lvl][1], win_size, blur_size)
        vx += new_vx
        vy += new_vy
        if lvl != 0:
            vx, vy = gaussian_expend(vx), gaussian_expend(vy)

    return vx, vy


if __name__ == '__main__':
    seq1, paths1 = load_folder("subject/DataSeq1")
    seq2, paths2 = load_folder("subject/DataSeq2")

    for i, (p, n) in enumerate(zip(seq1[:-1], seq1[1:])):
        u, v = hierarchical_lk(p, n, 4, 25)
        im = wrap(p, u, v)
        quiver(p, u, v, "Images/ps5-3-1-" + str(i) + "q.png")
        cv2.imwrite("Images/ps5-3-1-" + str(i) + ".png", int_img_to_unint8(p.astype(np.int) - im.astype(np.int)))

    for i, (p, n) in enumerate(zip(seq2[:-1], seq2[1:])):
        u, v = hierarchical_lk(p, n, 4, 25)
        im = wrap(p, u, v)
        quiver(p, u, v, "Images/ps5-3-2-" + str(i) + "q.png")
        cv2.imwrite("Images/ps5-3-2-" + str(i) + ".png", int_img_to_unint8(p.astype(np.int) - im.astype(np.int)))

    shift0 = cv2.imread("subject/TestSeq/Shift0.png", cv2.IMREAD_GRAYSCALE)

    for i in ['2', '10', '20', '40', '5U5']:
        shift_i = cv2.imread("subject/TestSeq/ShiftR"+i+".png", cv2.IMREAD_GRAYSCALE)
        u, v = hierarchical_lk(shift0, shift_i, 4, 21, None)
        im = wrap(shift0, u, v)
        quiver(shift0, u, v, "Images/ps5-3-0-" + i + "q.png")
        cv2.imwrite("Images/ps5-3-0-" + i + ".png", int_img_to_unint8(shift_i.astype(np.int)-im.astype(np.int)))
