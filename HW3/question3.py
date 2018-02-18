import cv2
import numpy as np

from HW3.question1 import compute_disparity2_mapped


if __name__ == '__main__':
    im1 = cv2.imread("subject/proj2-pair1-L.png", cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread("subject/proj2-pair1-R.png", cv2.IMREAD_GRAYSCALE)
    # im1_gb = cv2.GaussianBlur(im1, (11, 11), 0)
    im1_gn = im1 + np.random.normal(0, 100**0.5, im1.shape)
    win_size = 9
    d_min = 30
    d_max = 120
    disp_lr = compute_disparity2_mapped(im1_gn, im2,
                                        window_size=win_size,
                                        d_min=d_min,
                                        d_max=d_max,
                                        explore_right=False)
    disp_rl = compute_disparity2_mapped(im2, im1_gn,
                                        window_size=win_size,
                                        d_min=d_min,
                                        d_max=d_max,
                                        explore_right=True)
    cv2.imwrite("Images/ps2-3-a-0.png", im1_gn)
    cv2.imwrite("Images/ps2-3-a-1.png", (disp_lr - d_min) * 255 / (d_max - d_min))
    cv2.imwrite("Images/ps2-3-a-2.png", (disp_rl - d_min) * 255 / (d_max - d_min))
    im1_bc = np.array(im1, dtype=float)
    im1_bc *= 1.1
    im1_bc = im1_bc - (im1_bc % 255) * (im1 > 255)
    im1_bc = np.array(im1_bc, dtype=np.uint8)
    disp_lr = compute_disparity2_mapped(im1_bc, im2,
                                        window_size=win_size,
                                        d_min=d_min,
                                        d_max=d_max,
                                        explore_right=False)
    disp_rl = compute_disparity2_mapped(im2, im1_bc,
                                        window_size=win_size,
                                        d_min=d_min,
                                        d_max=d_max,
                                        explore_right=True)
    cv2.imwrite("Images/ps2-3-b-1.png", (disp_lr - d_min) * 255 / (d_max - d_min))
    cv2.imwrite("Images/ps2-3-b-2.png", (disp_rl - d_min) * 255 / (d_max - d_min))
