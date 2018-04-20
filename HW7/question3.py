import cv2
import numpy as np
from math import exp, ceil

from HW7.question1 import *


def update_patch(old_patch, current_patch, alpha=0.2):
    if old_patch.shape != current_patch.shape:
        old_patch = cv2.resize(old_patch, (current_patch.shape[1], current_patch.shape[0])).reshape(current_patch.shape)
    return alpha * old_patch + (1.0 - alpha) * current_patch


def predict_p(particles, center, old_center):
    last_move = center - old_center
    return particles + last_move // 2, last_move


def init_particles_s(count, u, v, noise_scale=10.0):
    p = np.random.normal((u, v, 0.0), noise_scale, size=(count, 3))
    p[:, 2] /= 100 * noise_scale
    p[:, 2] += 1.0
    return p


def update_ps(particles, img, ref, center, sigma=1000.0, noise_scale=10.0, threshold=10**6):
    aw, ah = tuple(map(lambda x: int(round(x//2)), ref.shape[0:2]))
    score_factor = -1.0 / (2.0 * sigma**2)
    scores = np.zeros(particles.shape[0])
    min_score = float('Inf')
    best_match = ref
    for i, (u, v, s) in enumerate(particles):
        try:
            u, v = int(u), int(v)
            paw, pah = int(round(s * aw)), int(round(s * ah))
            pred = img[u-paw:u+paw, v-pah:v+pah, :]
            pred_resize = cv2.resize(pred, (2 * ah, 2 * aw)).reshape(ref.shape)
            scores[i] = np.sum(np.power(pred_resize - ref, 2))
            if scores[i] < min_score:
                min_score = scores[i]
                best_match = pred
        except cv2.error:
            scores[i] = float('Inf')

    if min_score > threshold * ref.shape[0] * ref.shape[1]:
        # s = np.mean(particles[:, 2])
        # w, h = tuple(map(lambda x: int(2 * round(s * (x // 2))), ref.shape[0:2]))
        # ref = cv2.resize(ref, (h, w))
        p = particles + (np.random.normal(0.0, noise_scale, particles.shape)) / [1, 1, 100]
        return p, center, ref

    scores = np.exp(scores * score_factor)
    scores /= np.sum(scores)
    center = np.array(tuple(map(int, np.average(particles, axis=0, weights=scores))))
    indices = np.random.choice(len(particles), size=len(particles), replace=True, p=scores)
    p = particles[indices]
    p = p + (np.random.normal(0.0, noise_scale, p.shape)) / [1, 1, 5 * noise_scale] - [0, 0, 0.05]
    return p, center, best_match


def question3_1():
    video = cv2.VideoCapture("subject/pedestrians.avi")
    center = np.array((190, 260, 1.0))
    last_center = center
    move = np.array((0, 0))
    p = init_particles_s(1000, center[0], center[1], noise_scale=1.0)
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()
        if frame is None:
            break

        # Our operations on the frame come here
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).reshape(frame.shape[:2] + (1,))

        if i == 0:
            ref = get_patch(frame, 210, 50, int(center[0]), int(center[1]))
            cv2.imwrite("Images/ps6-3-1-0.png", ref)

        # p, move = predict_p(p, center, last_center)
        # last_center = center
        p, center, best_match = update_ps(p,
                                          frame,
                                          ref,
                                          center,
                                          noise_scale=1.5,
                                          sigma=600.0,
                                          threshold=1500 * 3)

        img = draw_particles(p, frame, ref.shape, center)

        ref = update_patch(ref, best_match, alpha=0.001)

        # Display the resulting frame
        cv2.imshow('frame', img)
        cv2.imshow('ref', ref.astype(np.uint8))

        if i in [40, 100, 240]:
            cv2.imwrite("Images/ps6-3-1-" + str(i) + ".png", img)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    question3_1()
