import cv2
import numpy as np
from multiprocessing import Pool
import itertools
import functools


def compute_disparity_new_mapped(img1, img2, window_size=3, d_min=0, d_max=100, explore_right=True):
    h, w = img1.shape
    win_offset = (window_size - 1) // 2
    w_min = win_offset
    w_max = w - win_offset
    h_min = win_offset
    h_max = h - win_offset
    if explore_right:
        w_max -= d_min
    else:
        w_min += d_min
        d_min, d_max = -d_max + 1, -d_min + 1

    all_yx = itertools.product(range(h_min, h_max), range(w_min, w_max))

    pool = Pool()

    disparity_1d = pool.map(functools.partial(explore_d,
                                              img1=img1,
                                              img2=img2,
                                              win_offset=win_offset,
                                              d_min=d_min,
                                              d_max=d_max),
                            all_yx)

    # disparity_1d = []
    # for yx in all_yx:
    #     disparity_1d.append(explore_d(yx,img1,img2,win_offset,d_min,d_max,w_max,w_min))

    target_w = w_max - w_min
    target_h = h_max - h_min
    disparity = np.fromiter(disparity_1d, dtype=int).reshape((target_h, target_w))
    return disparity


def explore_d(yx, img1, img2, win_offset, d_min, d_max):
    y, x = yx

    d_max, d_min = max(d_min, d_max), min(d_min, d_max)
    win1 = img1[y - win_offset:y + 1 + win_offset, x - win_offset:x + 1 + win_offset]
    win2 = img2[y - win_offset:y + 1 + win_offset, max(0, x - win_offset + d_min):x + 1 + win_offset + d_max]
    try:
        dist_array = cv2.matchTemplate(win2, win1, cv2.TM_CCOEFF_NORMED)
    except Exception as e:
        print(win1.shape, win2.shape, d_min, d_max, x, y, win1.dtype, win2.dtype)
        raise e
    return np.abs(dist_array.argmax() + min(d_min, d_max))


if __name__ == '__main__':
    im1 = cv2.imread("subject/proj2-pair1-L.png", cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread("subject/proj2-pair1-R.png", cv2.IMREAD_GRAYSCALE)
    win_size = 9
    min_d = 30
    max_d = 120
    disp_lr = compute_disparity_new_mapped(im1, im2,
                                           window_size=win_size,
                                           d_min=min_d,
                                           d_max=max_d,
                                           explore_right=False)
    disp_rl = compute_disparity_new_mapped(im2, im1,
                                           window_size=win_size,
                                           d_min=min_d,
                                           d_max=max_d,
                                           explore_right=True)
    cv2.imwrite("Images/ps2-4-a-1.png", (disp_lr - min_d) * 255 / (max_d - min_d))
    cv2.imwrite("Images/ps2-4-a-2.png", (disp_rl - min_d) * 255 / (max_d - min_d))

    # im1_gb = cv2.GaussianBlur(im1, (11, 11), 0)
    im1_gn = im1 + np.random.normal(0, 100 ** 0.5, im1.shape)
    im1_gn = im1_gn.astype(np.uint8)
    disp_lr = compute_disparity_new_mapped(im1_gn, im2,
                                           window_size=win_size,
                                           d_min=min_d,
                                           d_max=max_d,
                                           explore_right=False)
    disp_rl = compute_disparity_new_mapped(im2, im1_gn,
                                           window_size=win_size,
                                           d_min=min_d,
                                           d_max=max_d,
                                           explore_right=True)
    cv2.imwrite("Images/ps2-4-b-3.png", (disp_lr - min_d) * 255 / (max_d - min_d))
    cv2.imwrite("Images/ps2-4-b-4.png", (disp_rl - min_d) * 255 / (max_d - min_d))
    im1_bc = np.array(im1, dtype=float)
    im1_bc *= 1.1
    im1_bc = im1_bc - (im1_bc % 255) * (im1 > 255)
    im1_bc = np.array(im1_bc, dtype=np.uint8)
    disp_lr = compute_disparity_new_mapped(im1_bc, im2,
                                           window_size=win_size,
                                           d_min=min_d,
                                           d_max=max_d,
                                           explore_right=False)
    disp_rl = compute_disparity_new_mapped(im2, im1_bc,
                                           window_size=win_size,
                                           d_min=min_d,
                                           d_max=max_d,
                                           explore_right=True)
    cv2.imwrite("Images/ps2-4-b-5.png", (disp_lr - min_d) * 255 / (max_d - min_d))
    cv2.imwrite("Images/ps2-4-b-6.png", (disp_rl - min_d) * 255 / (max_d - min_d))

