import cv2
import numpy as np
from math import floor, atan2

from HW5.question1 import harris_transform


def compute_keypoint_from_harris(img:np.ndarray, keypoint_number=100, win_size=5):
    r, i_x, i_y = harris_transform(img, win_size)
    r_w, r_h = r.shape
    win_radius = int(floor((win_size - 1) / 2))
    kp_list = []
    for i in range(keypoint_number):
        # find good corner
        p = np.argmax(r)
        p_x, p_y = p // r_h, p % r_h
        # remove good corner neighbors
        r[p_x - win_radius:p_x + win_radius + 1, p_y - win_radius:p_y + win_radius + 1] = 0.0
        # create keypoint
        kp = cv2.KeyPoint()
        kp.angle = atan2(i_y[p_x, p_y], i_x[p_x, p_y]) * 180.0 / np.pi
        kp.size = 50.0
        kp.octave = 0
        kp.pt = (p_y, p_x)
        kp_list.append(kp)
    return kp_list


def extend_to_three_channels(one_chanel_img):
    ret = np.zeros(shape=one_chanel_img.shape + (3,), dtype=one_chanel_img.dtype)
    for i in range(3):
        ret[:, :, i] = one_chanel_img[:, :]
    return ret


def draw_matches(im1, im2, kp1, kp2, matches):
    h1, w1 = im1.shape
    h2, w2 = im2.shape
    ret = np.zeros(shape=(max(h1, h2), w1 + w2), dtype=np.uint8)
    ret[:h1, :w1] = im1
    ret[:h2, w1:] = im2
    ret = extend_to_three_channels(ret)
    for m in matches:
        x1, y1 = kp1[m[0].queryIdx].pt
        x2, y2 = kp2[m[0].trainIdx].pt
        ret = cv2.line(ret, (int(x1), int(y1)), (int(x2 + w1), int(y2)), (255, 0, 0))
    return ret


if __name__ == '__main__':
    trans_a = cv2.imread("subject/transA.jpg", cv2.IMREAD_GRAYSCALE)
    trans_b = cv2.imread("subject/transB.jpg", cv2.IMREAD_GRAYSCALE)
    sim_a = cv2.imread("subject/simA.jpg", cv2.IMREAD_GRAYSCALE)
    sim_b = cv2.imread("subject/simB.jpg", cv2.IMREAD_GRAYSCALE)
    im_list = [trans_a, trans_b, sim_a, sim_b]

    sift = cv2.xfeatures2d.SIFT_create()
    des = list()
    kp = list()

    for i, f in enumerate(im_list):
        kp.append(compute_keypoint_from_harris(f, 300, 5))
        # kp = sift.detect(f, None)
        f_kp = cv2.drawKeypoints(f, kp[i], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('Images/ps4-2-1-' + str(i) + '.png', f_kp)
        _, descriptor = sift.compute(trans_a, kp[i])
        des.append(descriptor)

    for i in range(0, 3, 2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des[i], des[i+1], 2)
        good = []
        for m, n in matches:
            if m.distance < n.distance * 0.75:
                good.append([m])
        img = draw_matches(im_list[i], im_list[i + 1], kp[i], kp[i + 1], good)
        # img = cv2.drawMatchesKnn(im_list[i], kp[i], im_list[i + 1], kp[i + 1], good, flags=2)
        cv2.imwrite('Images/ps4-2-2-' + str(i//2) + '.png', img)

