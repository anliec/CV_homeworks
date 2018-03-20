import cv2
import numpy as np
from math import floor, atan2
import sqlite3

from HW5.question2 import compute_keypoint_from_harris, draw_matches


def randsac_match_trans(kp1, kp2, number_of_try: int, max_dist: float):
    con = sqlite3.connect(":memory:")
    # con.execute("ATTACH DATABASE \":memory:\" AS memdb;")
    con.execute("CREATE TABLE kp1 (x INTEGER, y INTEGER);")
    con.execute("CREATE TABLE kp2 (x INTEGER, y INTEGER);")
    con.enable_load_extension(True)
    con.load_extension("./extension-functions")
    con.executemany("insert into kp1(x, y) values (?, ?)", map(lambda x: x.pt, kp1))
    con.executemany("insert into kp2(x, y) values (?, ?)", map(lambda x: x.pt, kp2))
    max_count = 0
    sq_max_dist = max_dist**2
    best_move = None
    best_matchs = []
    for i in range(number_of_try):
        p1 = np.random.choice(kp1)
        p2 = np.random.choice(kp2)
        movement = np.array(p2.pt) - np.array(p1.pt)
        matches = []
        for row in con.execute("SELECT kp1.rowid, kp2.rowid, "
                               "power(kp1.x+"+str(movement[0])+"-kp2.x, 2) + " 
                               "power(kp1.y+"+str(movement[1])+"-kp2.y, 2) as d "
                               " from kp1, kp2 where d <= "+str(sq_max_dist)):
            m = cv2.DMatch()
            m.queryIdx = int(row[0]) - 1
            m.trainIdx = int(row[1]) - 1
            m.distance = float(row[2])
            matches.append((m,))
        # for ip1 in kp1:
        #     for ip2 in kp2:
        #         diff = np.array(ip1.pt) + movement - np.array(ip2.pt)
        #         dist = np.sum(np.power(diff, 2))
        #         if dist <= sq_max_dist:
        #             count += 1
        #             matches.append((ip1, ip2))
        if len(matches) > max_count:
            max_count = len(matches)
            best_move = movement
            best_matchs = matches
    return best_matchs, best_move, max_count


def randsac_match_rot(kp1, kp2, number_of_try: int, max_dist: float):
    con = sqlite3.connect(":memory:")
    # con.execute("ATTACH DATABASE \":memory:\" AS memdb;")
    con.execute("CREATE TABLE kp1 (x INTEGER, y INTEGER);")
    con.execute("CREATE TABLE kp2 (x INTEGER, y INTEGER);")
    con.enable_load_extension(True)
    con.load_extension("./extension-functions")
    con.executemany("insert into kp1(x, y) values (?, ?)", map(lambda x: x.pt, kp1))
    con.executemany("insert into kp2(x, y) values (?, ?)", map(lambda x: x.pt, kp2))
    max_count = 0
    sq_max_dist = max_dist**2
    best_move = None
    best_matchs = []
    for i, row in enumerate(con.execute("SELECT k11.rowid, k12.rowid, k21.rowid, k22.rowid,"
                                        "power(k11.x-k12.x, 2) + "
                                        "power(k11.y-k12.y, 2) as d1, "
                                        "power(k21.x-k22.x, 2) + "
                                        "power(k21.y-k22.y, 2) as d2 "
                                        "from kp1 as k11, kp1 as k12, kp2 as k21, kp2 as 22 "
                                        "where abs(d1 - d2) <= 4 "
                                        "and k11.rowid != k12.rowid"
                                        "and k21.rowid != k22.rowid")):
        ik1 = (int(row[0]) - 1, int(row[1]) - 1)
        ik2 = (int(row[2]) - 1, int(row[3]) - 1)
        p1 = (kp1[ik1[0]], kp1[ik1[1]])
        p2 = (kp2[ik2[0]], kp2[ik2[1]])
        if (p1[0][0] - p1[1][0]) == 0 or (p2[0][0] - p2[1][0]):
            continue
        a1 = (p1[0][1] - p1[1][1]) / (p1[0][0] - p1[1][0])
        b1 = a1 * p1[0][0] + p1[0][1]
        a2 = (p2[0][1] - p2[1][1]) / (p2[0][0] - p2[1][0])
        b2 = a1 * p2[0][0] + p2[0][1]
        cr = np.zeros(2, np.int)
        cr[0] = b1 - b2 / (a1 - a2)
        cr[1] = a1 * cr[0] + b1
        angle = atan2(p2[0][1] - cr[1], p2[0][0] - cr[0]) - atan2(p1[0][1] - cr[1], p1[0][0] - cr[0])
        matches = []
        for row in con.execute("SELECT kp1.rowid, kp2.rowid, "
                               "power(kp1.x+"+str(movement[0])+"-kp2.x, 2) + " 
                               "power(kp1.y+"+str(movement[1])+"-kp2.y, 2) as d "
                               "from kp1, kp2 where d <= "+str(sq_max_dist)):
            m = cv2.DMatch()
            m.queryIdx = int(row[0]) - 1
            m.trainIdx = int(row[1]) - 1
            m.distance = float(row[2])
            matches.append((m,))
        # for ip1 in kp1:
        #     for ip2 in kp2:
        #         diff = np.array(ip1.pt) + movement - np.array(ip2.pt)
        #         dist = np.sum(np.power(diff, 2))
        #         if dist <= sq_max_dist:
        #             count += 1
        #             matches.append((ip1, ip2))
        if len(matches) > max_count:
            max_count = len(matches)
            best_move = movement
            best_matchs = matches
    return best_matchs, best_move, max_count


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
        _, descriptor = sift.compute(trans_a, kp[i])
        des.append(descriptor)

    for i in range(0, 3, 2):
        matches, move, count = randsac_match_trans(kp[i], kp[i + 1], 1000, 3)
        print(i, "->", len(matches), "matches for move", move)
        img = draw_matches(im_list[i], im_list[i + 1], kp[i], kp[i + 1], matches)
        cv2.imwrite('Images/ps4-3-1-' + str(i // 2) + '.png', img)



