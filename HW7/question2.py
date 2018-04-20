import cv2
import numpy as np
from math import exp

from HW7.question1 import *


def update_patch(old_patch, current_patch, alpha=0.2):
    return alpha * old_patch + (1.0 - alpha) * current_patch


def predict_p(particles, center, old_center):
    last_move = center - old_center
    return particles + last_move // 2, last_move


def question2_1():
    video = cv2.VideoCapture("subject/pres_debate.avi")
    center = np.array((460, 575))
    last_center = center
    move = np.array((0, 0))
    p = init_particles(600, center[0], center[1], noise_scale=1.0)
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()
        if frame is None:
            break

        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).reshape(frame.shape[:2] + (1,))

        if i == 0:
            ref = get_patch(frame, 100, 100, center[0], center[1])
            cv2.imwrite("Images/ps6-2-1-0.png", ref)

        p, move = predict_p(p, center, last_center)
        last_center = center
        p, center = update_p(p,
                             frame,
                             ref,
                             noise_scale=5.0 + np.linalg.norm(move, 2) * 1.0,
                             sigma=150.0)

        img = draw_particles(p, frame, ref.shape, center)

        ref = update_patch(ref, get_patch(frame, ref.shape[0], ref.shape[1], center[0], center[1]),
                           alpha=0.99)

        # Display the resulting frame
        cv2.imshow('frame', img)
        cv2.imshow('ref', ref.astype(np.uint8))

        if i in [15, 50, 140]:
            cv2.imwrite("Images/ps6-2-1-" + str(i) + ".png", img)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def question2_2():
    video = cv2.VideoCapture("subject/noisy_debate.avi")
    center = np.array((470, 575))
    last_center = center
    move = np.array((0, 0))
    p = init_particles(400, center[0], center[1], noise_scale=1.0)
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()
        if frame is None:
            break

        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).reshape(frame.shape[:2] + (1,))

        if i == 0:
            ref = get_patch(frame, 100, 100, center[0], center[1])
            cv2.imwrite("Images/ps6-2-2-0.png", ref)

        p, move = predict_p(p, center, last_center)
        last_center = center
        p, center = update_p(p,
                             frame,
                             ref,
                             noise_scale=5.0 + np.linalg.norm(move, 2) * 1.0,
                             sigma=150.0)

        img = draw_particles(p, frame, ref.shape, center)

        ref = update_patch(ref,
                           get_patch(frame, ref.shape[0], ref.shape[1], center[0], center[1]),
                           alpha=0.85)

        # Display the resulting frame
        cv2.imshow('frame', img)

        if i in [15, 50, 140]:
            cv2.imwrite("Images/ps6-2-2-" + str(i) + ".png", img)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    question2_2()
