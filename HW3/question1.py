import numpy as np
import cv2
import itertools
import time
from multiprocessing import Pool
from functools import partial


def compute_disparity(img1, img2, window_size=3, d_min=0, d_max=100, explore_right=True):
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

    disparity = np.zeros((h_max - h_min, w_max - w_min), dtype=int)

    for x_index, x in enumerate(range(w_min, w_max)):
        for y_index, y in enumerate(range(h_min, h_max)):
            min_diff = None
            dist = None
            xd_min = x + d_min
            xd_max = x + d_max
            if explore_right:
                xd_max = min([xd_max, w_max])
            else:
                xd_min = max([xd_min, w_min])
            for xd_index, xd in enumerate(range(xd_min, xd_max)):
                win1 = img1[y - win_offset:y + win_offset + 1, x - win_offset:x + win_offset + 1]
                win2 = img2[y - win_offset:y + win_offset + 1, xd - win_offset:xd + win_offset + 1]
                diff = ssd(win1, win2)
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    dist = xd - x
            disparity[y_index, x_index] = abs(dist)
    return disparity


# add multi process: https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python
# and https://stackoverflow.com/questions/9911819/python-parallel-map-multiprocessing-pool-map-with-global-data
def compute_disparity_mapped(img1, img2, window_size=3, d_min=0, d_max=100, explore_right=True):
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

    disparity_1d = pool.map(partial(explore_d,
                                    img1=img1,
                                    img2=img2,
                                    win_offset=win_offset,
                                    d_min=d_min,
                                    d_max=d_max,
                                    w_max=w_max,
                                    w_min=w_min),
                            all_yx)

    target_w = w_max - w_min
    target_h = h_max - h_min
    disparity = np.fromiter(disparity_1d, dtype=int).reshape((target_h, target_w))
    return disparity


def ssd(window1, window2):
    return np.sum(np.power(window1.astype(np.int) - window2.astype(np.int), 2))


def explore_d(yx, img1, img2, win_offset, d_min, d_max, w_max, w_min):
    y, x = yx

    def comp_windows(d):
        if x + d >= w_max:
            return np.iinfo(np.int).max
        win1 = img1[y - win_offset:y + 1 + win_offset, x - win_offset:x + 1 + win_offset]
        win2 = img2[y - win_offset:y + 1 + win_offset, x - win_offset + d:x + 1 + win_offset + d]
        return ssd(win1, win2)
    cur_d_max = min([d_max, w_max - x + 1])
    cur_d_min = max([d_min, w_min - x])
    disparity_list = map(comp_windows, range(cur_d_min, cur_d_max))
    return np.abs(np.fromiter(disparity_list, dtype=int).argmin() + cur_d_min)


def compute_disparity2(img1, img2, window_size=3, d_min=0, d_max=100, explore_right=True):
    h, w = img1.shape

    border_size = d_max + 1
    img1 = cv2.copyMakeBorder(img1, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)
    img2 = cv2.copyMakeBorder(img2, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)

    win_offset = (window_size - 1) // 2
    w_min = border_size
    w_max = border_size + w
    h_min = border_size
    h_max = border_size + h
    d_min_abs = d_min
    if explore_right:
        w_max -= d_min
    else:
        w_min += d_min
        d_min, d_max = -d_max + 1, -d_min + 1

    target_w = w_max - w_min - 2 * win_offset + d_min_abs
    target_h = h_max - h_min - 2 * win_offset
    disparity_val = np.zeros((d_max - d_min, target_h, target_w),
                             dtype=int)

    for d_index, d in enumerate(range(d_min, d_max)):
        img1_crop = img1[h_min:h_max, w_min:w_max]
        img2_crop = img2[h_min:h_max, w_min + d:w_max + d]
        sd = img1_crop.astype(np.int) - img2_crop.astype(np.int)
        sd = np.power(sd, 2)

        for x_index, x in enumerate(range(target_w)):
            for y_index, y in enumerate(range(target_h)):
                win = sd[y:y + 2*win_offset + 1, x:x + 2*win_offset + 1]
                disparity_val[d_index, y_index, x_index] = np.sum(win)

    ret = np.argmin(disparity_val, axis=0) + d_min
    return np.abs(ret)


def compute_disparity2_mapped(img1, img2, window_size=3, d_min=0, d_max=100, explore_right=True):
    h, w = img1.shape

    border_size = d_max + 1
    img1 = cv2.copyMakeBorder(img1, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)
    img2 = cv2.copyMakeBorder(img2, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)

    win_offset = (window_size - 1) // 2
    w_min = border_size
    w_max = border_size + w
    h_min = border_size
    h_max = border_size + h
    d_min_abs = d_min
    if explore_right:
        w_max -= d_min
    else:
        w_min += d_min
        d_min, d_max = -d_max + 1, -d_min + 1

    arg_list = []

    for d_index, d in enumerate(range(d_min, d_max)):
        img1_crop = img1[h_min:h_max, w_min:w_max]
        img2_crop = img2[h_min:h_max, w_min + d:w_max + d]
        arg_list.append((img1_crop, img2_crop, win_offset, d_min_abs))

    pool = Pool()
    disparity_plan_iter = pool.map(compute_d_plan, arg_list)
    ret = np.argmin(disparity_plan_iter, axis=0) + d_min
    return np.abs(ret)


def compute_d_plan(info):
    img1, img2, win_offset, d_min = info
    target_h, target_w = img1.shape
    target_w -= 2 * win_offset + d_min
    target_h -= 2 * win_offset
    sd = img1.astype(np.int) - img2.astype(np.int)
    sd = np.power(sd, 2)

    disparity_plan = np.zeros((target_h, target_w), dtype=int)
    for x_index, x in enumerate(range(target_w)):
        for y_index, y in enumerate(range(target_h)):
            win = sd[y:y + 2 * win_offset + 1, x:x + 2 * win_offset + 1]
            disparity_plan[y_index, x_index] = np.sum(win)

    return disparity_plan


if __name__ == '__main__':
    im1 = cv2.imread("subject/leftTest.png", cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread("subject/rightTest.png", cv2.IMREAD_GRAYSCALE)

    start = time.time()
    # disparity = compute_disparity_mapped(im1, im2,
    #                                      window_size=11,
    #                                      d_min=0,
    #                                      d_max=10)
    disp = compute_disparity2(im1, im2,
                              window_size=11,
                              d_min=0,
                              d_max=10)
    print(time.time() - start)
    print(disp.shape)
    print(disp.dtype)
    print(np.max(disp))
    print(np.min(disp))

