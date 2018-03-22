import cv2
import numpy as np
from math import floor, atan2
import sqlite3

from HW5.question2 import compute_keypoint_from_harris, draw_matches


def randsac_match_trans(kp1, kp2, matches, number_of_try: int, max_dist: float):
    matches = list(map(lambda x: x[0], matches))
    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)
    con.load_extension("./extension-functions")
    con.execute("CREATE TABLE m (p1x INTEGER, p1y INTEGER, p2x INTEGER, p2y INTEGER);")
    con.executemany("insert into m(p1x, p1y, p2x, p2y) values (?, ?, ?, ?)",
                    map(lambda x: kp1[x.queryIdx].pt + kp2[x.trainIdx].pt, matches))
    max_count = 0
    sq_max_dist = max_dist**2
    best_move = None
    best_matchs = []
    for i in range(number_of_try):
        m = np.random.choice(matches)
        p1, p2 = kp1[m.queryIdx], kp2[m.trainIdx]
        movement = np.array(p2.pt) - np.array(p1.pt)
        matches_list = []
        for row in con.execute("SELECT m.rowid "
                               # "power(m.p1x + "+str(movement[0])+" - m.p2x, 2) + " 
                               # "power(m.p1y + "+str(movement[1])+" - m.p2y, 2) as d "
                               # "from m where d <= "+str(sq_max_dist)+";"):
                               "FROM m "
                               "WHERE abs(m.p1x + "+str(movement[0])+" - m.p2x) <= "+str(sq_max_dist)+" "
                               "AND abs(m.p1y + "+str(movement[1])+" - m.p2y) <= "+str(sq_max_dist)+";"):
            matches_list.append((matches[row[0] - 1],))
        if len(matches_list) > max_count:
            max_count = len(matches_list)
            best_move = movement
            best_matchs = matches_list
    con.execute("DROP TABLE m")
    con.close()
    return best_matchs, best_move, max_count


def randsac_match_affine(kp1, kp2, matches, number_of_try: int, max_dist: float):
    matches = list(map(lambda x: x[0], matches))
    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)
    con.load_extension("./extension-functions")
    con.execute("CREATE TABLE m (p1x INTEGER, p1y INTEGER, p2x INTEGER, p2y INTEGER);")
    con.executemany("insert into m(p1x, p1y, p2x, p2y) values (?, ?, ?, ?)",
                    map(lambda x: kp1[x.queryIdx].pt + kp2[x.trainIdx].pt, matches))
    max_count = 0
    best_move = None
    best_matchs = []
    for i in range(number_of_try):
        while True:
            m = np.random.choice(matches, 2, replace=False)
            p1, p2 = list(), list()
            for mi in m:
                p1.append(np.array(kp1[mi.queryIdx].pt))
                p2.append(np.array(kp2[mi.trainIdx].pt))
            if np.sum(p1[0] == p1[1]) != 2 and np.sum(p2[0] == p2[1]) != 2:
                break
        m = np.mat(
            [
                [p2[0][0], -p2[0][1], 1, 0],
                [p2[0][1],  p2[0][0], 0, 1],
                [p2[1][0], -p2[1][1], 1, 0],
                [p2[1][1],  p2[1][0], 0, 1]
            ]
        )
        v = np.mat(
            [
                [p1[0][0]],
                [p1[0][1]],
                [p1[1][0]],
                [p1[1][1]]
            ]
        )
        t = np.array(np.linalg.inv(m) * v)
        a, b, c, d = [t[i][0] for i in range(4)]
        # compute inverse transformation
        det = 1 / (a**2 + b**2)
        ai = a * det
        bi = -b * det
        ci = -(a * c + b * d) * det
        di = (b * c - a * d) * det
        matches_list = []
        for row in con.execute("SELECT m.rowid, "
                               "" + str(a) + " * m.p2x - " + str(b) + " * m.p2y +" + str(c) + " as u,"
                               "" + str(b) + " * m.p2x + " + str(a) + " * m.p2y +" + str(d) + " as v,"
                               "" + str(ai) + " * m.p1x - " + str(bi) + " * m.p1y +" + str(ci) + " as ui,"
                               "" + str(bi) + " * m.p1x + " + str(ai) + " * m.p1y +" + str(di) + " as vi "
                               "FROM m "
                               # "WHERE power(kp1.x-u, 2) + power(kp1.y-v, 2) <= " + str(sq_max_dist) + " "
                               # "AND power(m.p2x-ui, 2) + power(m.p2y-vi, 2) <= " + str(sq_max_dist) + ";"):
                               "WHERE abs(m.p1x-u)<=" + str(max_dist) + " AND abs(m.p1y-v) <= " + str(max_dist) + " "
                               "AND abs(m.p2x-ui)<=" + str(max_dist) + " AND abs(m.p2y-vi)<=" + str(max_dist) + ";"):
            matches_list.append((matches[row[0] - 1],))
        if len(matches_list) > max_count:
            max_count = len(matches_list)
            best_move = t
            best_matchs = matches_list
    con.execute("DROP TABLE m")
    con.close()
    return best_matchs, best_move, max_count


if __name__ == '__main__':
    trans_a = cv2.imread("subject/transA.jpg", cv2.IMREAD_GRAYSCALE)
    trans_b = cv2.imread("subject/transB.jpg", cv2.IMREAD_GRAYSCALE)
    sim_a = cv2.imread("subject/simA.jpg", cv2.IMREAD_GRAYSCALE)
    sim_b = cv2.imread("subject/simB.jpg", cv2.IMREAD_GRAYSCALE)
    im_list = [trans_a, trans_b, sim_a, sim_b]

    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    kp = list()
    des = list()

    for i, f in enumerate(im_list):
        kp.append(compute_keypoint_from_harris(cv2.GaussianBlur(f, (9, 9), 0), 400, 10))
        _, descriptor = sift.compute(f, kp[i])
        des.append(descriptor)

    for i in range(0, 3, 2):
        matches = bf.knnMatch(des[i], des[i + 1], k=1)
        matches, move, count = randsac_match_trans(kp[i], kp[i + 1], matches, 3000, 4)
        print(i, "->", len(matches), "matches for move", move)
        img = draw_matches(im_list[i], im_list[i + 1], kp[i], kp[i + 1], matches)
        cv2.imwrite('Images/ps4-3-1-' + str(i // 2 + 1) + '.png', img)

    for i in range(0, 3, 2):
        matches = bf.knnMatch(des[i], des[i + 1], k=1)
        matches, move, count = randsac_match_affine(kp[i], kp[i + 1], matches, 3000, 4)
        print(i, "->", len(matches), "matches for transform\n", move)
        img = draw_matches(im_list[i], im_list[i + 1], kp[i], kp[i + 1], matches)
        cv2.imwrite('Images/ps4-3-2-' + str(i // 2 + 1) + '.png', img)


