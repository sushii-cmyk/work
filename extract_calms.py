import json
import os.path
from os.path import join
from sys import float_repr_style
import msvcrt

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.ma.core import shape

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

def coords(part, id, ants):
    """ ants = # (frames) x (mouse ID) x (x, y coordinate) x (body part) """
    u = bp_ids.index(part)
    ps = ants[:, id, :, u]

    return ps

def parse_diff(part, id, qart, jd, ants=None):
    """
    returns the displacement vectors
    (first body part coords) - (second body part coods). These vectors
    point from Qart to Part. CalMS bodyparts go:
    "nose", "earl", "earr", "neck", "hipl", "hipr", "tail"

    :param part: menuend mouse bodypart
    :type part: str
    :param id: menuend mouse id
    :type id: int
    :param qart: subtrahend mouse bodypart
    :type qart: str
    :param jd: subtrahend mouse id
    :type jd: int
    :return: array
    :rtype: np.array
    """

    for p in [part, qart]:
        if p not in bp_ids:
            raise KeyError(f"No such bodypart id \'{p}\'."
                           f"Known bodyparts are: {', '.join(bp_ids)}.")

    ps = coords(part, id, ants)
    qs = coords(qart, jd, ants)
    rs = ps - qs
    # print("P", ps, "Q", qs, "R", rs)

    return rs


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
        frame[q[0], q[1]] = 255 * np.sqrt(array([1 - i / n_, 0, i / n_]))

    return frame


# !!!!!!!! here I, J = MOUSE ID, BODYPART ID
def circ_an(ants, i, j, r, a = 0, b = 1):
    ts = unit(tau * r, a, b)
    c = (vunc(rect) * (r * ts)).T

    a = np.full(np.shape(c), ants[i].T[j])

    return c + a


def circ_ij(i, j, r, a = 0., b = 1.):
    ts = unit(tau * r, 1 - a, 1 - b)
    # print(exp2ni(ts))
    return complex(i, j) + r * ts

def diff(ants, i, j):
    return ants[i] - ants[j]


def main():
    # mouse id
    m_id = 73

    # file for annotations, annotations as points
    video, ants = files(m_id)

    # lines
    a0 = draw_n_frames(video, ants, 0, n=10)
    a = a0

    # diff tests
    n_ = 150
    ns_01 = parse_diff("nose", 0, "nose", 1, ants)[0]
    ns__01 = uint(n_).T @ (ns_01.reshape((1, 2)))
    print(ns__01)
    ns___01 = coords("nose", 0, ants)[0]
    print(ns___01)
    ns_ = np.apply_along_axis(lambda x: ns___01 - x, 1, ns__01)
    print(ns_)
    ns__ = (rovnd * ns_).T
    print(ns__)

    a[ns__[1], ns__[0]] = [0, 0, 255]


    # circle!
    circ = circ_ij(300, 300, 10)
    xirc = rovnd * (vunc(rect) * circ.T)
    a[xirc[1], xirc[0]] = [255, 0, 0]

    # another circle!
    dirc = circ_ij(400, 400, 20, 0, 0.3)
    yirc = rovnd * (vunc(rect) * dirc.T)
    a[yirc[1], yirc[0]] = [0, 255, 0]

    plt.imshow(a)
    plt.show()

    cv2.destroyAllWindows()

def nain():# mouse id
    m_id = 73
    # file for annotations, annotations as points
    video, ants = files(m_id)

    n = 0
    for i in range(1000):
        g = msvcrt.getch()
        if g == b'q':
            quit()

        if g == b'w':
            n += 1
            if n > 9:
                break
    else:
        on = False

if __name__ == '__main__':
    nain()

    # print(f(lambda x: x) * [1, 2])

    quit()
