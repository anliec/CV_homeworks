import numpy as np
import cv2
import itertools


def compute_disparity(img1, img2, window_size=3, d_min=0, d_max=100):
    w, h = img1.shape
    win_offset = (window_size - 1) // 2
    w_min = win_offset
    w_max = w - win_offset
    h_min = win_offset
    h_max = h - win_offset

    disparity = np.zeros((h, w))

    for x in range(w_min, w_max):
        for y in range(h_min, h_max):
            min_dist = 99999
            for d in range(d_min, d_max):
                win1 = img1[x - win_offset:x + win_offset, y - win_offset:y + win_offset]
                win2 = img2[x - win_offset + d:x + win_offset + d, y - win_offset:y + win_offset]
                dist = ssd(win1, win2)
                min_dist = min([min_dist, dist])
            disparity[x, y] = min_dist


def compute_disparity_mapped(img1, img2, window_size=3, d_min=0, d_max=100):
    w, h = img1.shape
    win_offset = (window_size - 1) // 2
    w_min = win_offset
    w_max = w - win_offset
    h_min = win_offset
    h_max = h - win_offset

    all_x, all_y = zip(*itertools.product(range(w_min, w_max), range(h_min, h_max)))

    def explore_d(y, x):
        def comp_windows(d):
            win1 = img1[x - win_offset:x + win_offset, y - win_offset:y + win_offset]
            win2 = img2[x - win_offset + d:x + win_offset + d, y - win_offset:y + win_offset]
            return ssd(win1, win2)
        disparity_list = map(comp_windows, range(d_min, d_max))
        return min(disparity_list)

    disparity_1d = map(explore_d, all_y, all_x)

    disparity = np.array(disparity_1d)
    disparity.reshape(shape=(w, h))
    return disparity


def ssd(window1, window2):
    return np.sum(np.power(window1 - window2, 2))


