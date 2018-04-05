import cv2
import numpy as np


def gaussian_reduce(im: np.ndarray):
    im = cv2.GaussianBlur(im, (3, 3), 0)
    im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    return im


def gaussian_expend(im: np.ndarray):
    im = cv2.resize(im, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    im = cv2.GaussianBlur(im, (3, 3), 0)
    return im


def laplacian_reduce(im: np.ndarray):
    r = gaussian_reduce(im.copy())
    e = gaussian_expend(r.copy())
    im = cv2.resize(im, (e.shape[1], e.shape[0]))
    return r, im.astype(np.int) - e.astype(np.int)


if __name__ == '__main__':
    seq1 = cv2.imread("subject/DataSeq1/yos_img_01.jpg", cv2.IMREAD_GRAYSCALE)

    img = seq1
    cv2.imwrite("Images/ps5-1-1-0.png", img)
    for i in range(4):
        img, error = laplacian_reduce(img)
        error += 128
        cv2.imwrite("Images/ps5-1-1-" + str(i + 1) + ".png", img.astype(np.uint8))
        cv2.imwrite("Images/ps5-1-2-" + str(i + 1) + ".png", error.astype(np.uint8))

