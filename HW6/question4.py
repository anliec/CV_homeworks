import cv2
import numpy as np

from HW6.question3 import hierarchical_lk
from HW6.question2 import wrap, load_folder, quiver
from HW5.question1 import int_img_to_unint8


if __name__ == '__main__':
    seq, paths = load_folder("subject/Juggle")

    for i, (p, n) in enumerate(zip(seq[:-1], seq[1:])):
        u, v = hierarchical_lk(p, n,
                               level=4,
                               win_size=50,
                               blur_size=None)
        im = wrap(n, u, v)
        quiver(p, u, v, "Images/ps5-4-1-" + str(i) + "q.png", scale=0.1)
        cv2.imwrite("Images/ps5-4-1-" + str(i) + ".png", int_img_to_unint8(p.astype(np.int) - im.astype(np.int)))
