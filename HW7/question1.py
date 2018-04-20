import cv2
import numpy as np
from math import exp


def get_patch(image, w, h, u, v):
    aw = w // 2
    ah = h // 2
    return image[u-aw:u+aw, v-ah:v+ah, :]


def init_particles(count, u, v, noise_scale=10.0):
    p = np.random.normal((u, v), noise_scale, size=(count, 2))
    return p.astype(np.int)


def update_p(particles, img, ref, sigma=1000.0, noise_scale=10.0):
    aw, ah = tuple(map(lambda x: int(x//2), ref.shape[0:2]))
    score_factor = -1.0 / (2.0 * sigma**2)
    scores = np.zeros(particles.shape[0])
    for i, (u, v) in enumerate(particles):
        try:
            pred = img[u-aw:u+aw, v-ah:v+ah, :]
            scores[i] = np.sum(np.power(pred - ref, 2))
        except ValueError:
            scores[i] = float('Inf')

    scores = np.exp(scores * score_factor)
    scores /= np.sum(scores)
    center = np.array(tuple(map(int, np.average(particles, axis=0, weights=scores))))
    indices = np.random.choice(len(particles), size=len(particles), replace=True, p=scores)
    p = particles[indices]
    p = (p.astype(np.float64) + np.random.normal(0.0, noise_scale, p.shape)).astype(np.int)
    return p, center


def draw_particles(particles, image, patch_shape, center=None):
    i = image.copy()
    for u, v in particles[:, :2]:
        cv2.circle(i, (int(v), int(u)), 2, (255, 0, 0))

    if center is None:
        center_u, center_v = tuple(map(int, np.average(particles, axis=0)))
    else:
        center_u, center_v = center[:2]
    p_aw, p_ah = tuple(map(lambda x: x // 2, patch_shape[0:2]))

    deviation = int(np.mean(np.linalg.norm(particles - center, axis=1)))

    cv2.circle(i, (center_v, center_u), deviation, (0, 255, 0))
    cv2.rectangle(i, (center_v - p_ah, center_u - p_aw), (center_v + p_ah, center_u + p_aw),
                  (0, 255, 0), 2)
    return i


def question1_1():
    video = cv2.VideoCapture("subject/pres_debate.avi")
    p = init_particles(10, 255, 370, noise_scale=1.0)
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()
        if frame is None:
            break

        if i == 0:
            ref = get_patch(frame, 140, 120, 255, 370)
            cv2.imwrite("Images/ps6-1-1-0.png", ref)

        p, center = update_p(p, frame, ref, noise_scale=7.0, sigma=100)

        img = draw_particles(p, frame, ref.shape, center)

        # Display the resulting frame
        cv2.imshow('frame', img)

        if i in [28, 84, 144]:
            cv2.imwrite("Images/ps6-1-1-" + str(i) + ".png", img)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def question1_5():
    video = cv2.VideoCapture("subject/noisy_debate.avi")
    p = init_particles(50, 255, 370, noise_scale=1.0)
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i == 0:
            ref = get_patch(frame, 120, 110, 255, 370)
            cv2.imwrite("Images/ps6-1-2-0.png", ref)

        p, center = update_p(p, frame, ref, noise_scale=3.0, sigma=100)

        img = draw_particles(p, frame, ref.shape, center)

        # Display the resulting frame
        cv2.imshow('frame', img)

        if i in [14, 32, 46]:
            cv2.imwrite("Images/ps6-1-5-" + str(i) + ".png", img)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    question1_5()
