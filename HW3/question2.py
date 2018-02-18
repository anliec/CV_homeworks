import cv2

from HW3.question1 import compute_disparity2_mapped


if __name__ == '__main__':
    im1 = cv2.imread("subject/proj2-pair1-L.png", cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread("subject/proj2-pair1-R.png", cv2.IMREAD_GRAYSCALE)
    win_size = 9
    d_min = 30
    d_max = 120
    disp_lr = compute_disparity2_mapped(im1, im2,
                                        window_size=win_size,
                                        d_min=d_min,
                                        d_max=d_max,
                                        explore_right=False)
    disp_rl = compute_disparity2_mapped(im2, im1,
                                        window_size=win_size,
                                        d_min=d_min,
                                        d_max=d_max,
                                        explore_right=True)
    cv2.imwrite("Images/ps2-2-a-2.png", (disp_lr - d_min) * 255 / (d_max - d_min))
    cv2.imwrite("Images/ps2-2-a-1.png", (disp_rl - d_min) * 255 / (d_max - d_min))
