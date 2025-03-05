import json
import os.path
from os.path import join
from sys import float_repr_style

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import *

R = slice(None, None, -1)


class view:
    def __init__(self, video, start, width):
        self.vid_path = video
        self.start = start
        self.width = width

        self.pos = self.start
        self.vid = cv2.VideoCapture(self.vid_path)

    def get(self):
        success, image = self.vid.read()
        if success:
            self.pos += 1
            return image
        else:
            return None

    def get_n(self, n):
        r = []
        while p := self.get() and (n := n - 1):
            r.append(p)

        return r

log_files = logger("F>")
log_debug = logger("D>")

bv_ids = {
    0: "attack",
    1: "investigation",
    2: "mount",
    3: "other"
}

# l + r bps might eb switched
bp_ids = ["nose", "earl", "earr", "neck", "hipl", "hipr", "tail"]

def diffs(x):
    # all differences (i < j)
    r = []

    for i, xi in enumerate(x[:-1]):
        s = []  # differences from xi
        for j, xj in enumerate(x[i + 1:]):
            s.append(array(x[i + j + 1]) - array(x[i]))
        else:
            r.append(s)

    log_debug[r]

    return r


def id(x):
    return str(x).rjust(3, "0")


# mouse = i;  < 71 in training
# annotator = j;  < 71 in training
def files(i=1):
    if i < 71:
        sub = "train"
    else:
        sub = "test"

    # paths
    abs = r"C://Users/danie/Desktop/code/work/CalMS21/"
    vid = f"task1_videos_mp4/task1_videos_mp4/{sub}/mouse{id(i)}_task1_annotator1.mp4"
    ant = f"task1_classic_classification/calms21_task1_{sub}.json"
    # output subdir
    out = f"keypoints"
    end = lambda j: f"t1-m{id(i)}-a{j}.json"
    # annotator id (as per video, not file)
    a_id = 1
    # json keys in order
    j = f"annotator-id_{a_id - 1}"
    k = f"task1/test/mouse{id(i)}_task1_annotator{a_id}"

    kptf = join(abs, out, end(a_id))

    # use saved version
    if os.path.isfile(kptf):
        log_files[f"Loaded {kptf}"]
        with open(kptf) as f:
            kpts = json.load(f)
    # otherwise, load + save new kpnt file
    else:
        load = join(abs, ant)
        log_files[f"Loaded {load}"]
        with open(load) as f:
            """ kpts = # (frames) x (mouse ID) x (x, y coordinate) x (body part) """
            kpts = json.load(f)[j][k]["keypoints"]

        log_files[f"Saved {kptf}"]
        with open(kptf, "w") as f:
            json.dump(kpts, f, indent=4)

    return abs + vid, array(kpts)


def get_frame(n, vid):
    vidcap = cv2.VideoCapture(vid)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, n)
    success, image = vidcap.read()
    if success:
        return image
    else:
        return None


# n = frame #, i = mouse id, vid = path, ant = json
def at_frame(n, i, vid, ant):
    antns = array(ant[n])
    frame = get_frame(n, vid)
    idx = rovnd[antns[i][R]]

    return frame, idx


def vel(kpts):
    i = 0  # 0 = black, 1 = white
    p = 0  # bodypart

    t = 3  # threshold
    kpts = kpts[:t]

    log_debug[diffs(kpts)]

    vel = diffs(kpts)
    x = range(len(vel))

    log_debug[vel]
    # vel = np.apply_along_axis(np.linalg.norm, 1, vel)
    # log_debug[vel]
    #
    # plt.plot(x, vel)
    # plt.show()

    return vel

def draw_frame(video, ants):
    frame, pnts = at_frame(0, 1, video, ants)
    fram2, pnt2 = at_frame(1, 1, video, ants)

    print(np.shape(pnts))
    for i, p in enumerate(pnts.T):
        # place body parts
        frame[p[0], p[1]] = [255, 0, 40 * i]

    for i, p in enumerate(np.linspace(pnts, pnt2, 10)):
        q = np.vectorize(int)(p)
        frame[q[0], q[1]] = [0, 255, 0]


    Image.fromarray(frame).show()

def draw_n_frames(video, ants, a=0, base=None, n=5):
    data = [at_frame(i, 1, video, ants) for i in range(a, n)]
    frames = array([d[0] for d in data])
    annots = array([d[1] for d in data])

    frame, annot = frames[0], annots[0]
    if base:
        frame = base

    bnnots = array([a - annot for a in annots])
    mean = 1 / n * sum(bnnots)

    n_ = 15
    s_ = 2.5
    for i, p in enumerate(np.linspace(0, s_ * mean, n_)):
        q = np.vectorize(int)(annot + p)
        frame[q[0], q[1]] = 255 * array([i / n_, 0, 1 - i / n_])

    # draw paths
    # for t, (f, g) in enumerate(frames):
    #     for i, p in enumerate(g.T):
    #         # place body parts
    #         frame[p[0], p[1]] = [255, 0, 40 * i]
    #
    #     frame[q[0]]
    #
    #     for i, p in enumerate(np.linspace(h, g, n)):
    #         q = np.vectorize(int)(p)
    #         k = int(255 * i / n)
    #
    #         frame[q[0], q[1]] = (0, 255 * i / n, 0)
    #     h = g

    # d0 = diff(pnts - frames[0][1], 0, 1)
    # d1 = diff(pnts - frames[1][1], 0, 1)
    # d01 = diff(array(frames[0][1]) - frames[1][1], 0, 1)
    # print(d0, d1, d01)

    return frame

# !!!!!!!! here I, J = MOUSE ID, BODYPART ID
def circ_an(ants, i, j, r):
    ts = unit(tau * r)
    return ants[i][j] + r * ts

def circ_ij(i, j, r):
    ts = unit(tau * r)
    # print(exp2ni(ts))
    return complex(i, j) + r * ts

def diff(ants, i, j):
    return ants[i] - ants[j]


def main():
    # mouse id
    m_id = 73

    # file for annotations, annotations as point s
    video, ants = files(m_id)

    log_debug[np.shape(video), np.shape(ants)]
    a0 = draw_n_frames(video, ants, 0, n=10)
    # a1 = draw_n_frames(video, ants, 10, n=1, base = a0)

    circ = circ_ij(200, 200, 20)

    a = a0
    # c = [x y, x y, x y...]
    # c.T = [x x x..., y y y...]
    # c = vunc(rect)[c]
    # print(vunc(lambda x: x)[c])
    for c in circ:
        x = rect(c)
        # print(">>", rect(x))
        a[*rovnd[x]] = [255, 0, 255]
    # print(c)

    xirc = vunc(rect)[circ]
    print(xirc)
    a[xirc[0], xirc[1]] = [255, 255, 0]

    plt.imshow(a)
    plt.show()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

    # print(f(lambda x: x) * [1, 2])


    quit()
