import json
import os.path
import sys
import time
from os.path import join
import msvcrt
from threading import Event

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pandas.core.common import any_none
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.ndimage import *
from utils import *
from videofig import videofig

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

basic_plot = lambda data, **flags: plt.plot(range(len(data)), data, **flags)


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

    abs = r"C:/Users/danie/code/work/CalMS21/"
    vid = f"task1_videos_mp4/task1_videos_mp4/{sub}/mouse{id(i)}_task1_annotator1.mp4"
    ant = f"task1_classic_classification/calms21_task1_{sub}.json"
    # output subdir
    out = f"keypoints"
    end = lambda j: f"t1-m{id(i)}-a{j}.json"
    # annotator id (as per video, not file)
    a_id = 1
    # json keys in order
    j = f"annotator-id_{a_id - 1}"
    k = f"task1/{sub}/mouse{id(i)}_task1_annotator{a_id}"

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


def get_video(vid):
    vidcap = cv2.VideoCapture(vid)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return vidcap


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

    # print(np.shape(pnts))
    for i, p in enumerate(pnts.T):
        # place body parts
        frame[p[0], p[1]] = [255, 0, 40 * i]

    for i, p in enumerate(np.linspace(pnts, pnt2, 10)):
        q = np.vectorize(int)(p)
        frame[q[0], q[1]] = [0, 255, 0]

    Image.fromarray(frame).show()


def coords(part, id, ants):
    """ ants = # (frames) x (mouse ID) x (x, y coordinate) x (body part) """
    u = bp_ids.index(part) if part in bp_ids else part
    ps = ants[:, id, :, u]

    return ps


def parse_diff(part, id, qart, jd, ants=None):
    """
    returns the displacement vectors
    (first body part coords) - (second body part coods). These vectors
    point from Qart to Part. CalMS bodyparts go:
    "nose", "earl", "earr", "neck", "hipl", "hipr", "tail"

    :param part: minuend mouse bodypart
    :type part: str
    :param id: minuend mouse id
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
def circ_an(ants, i, j, r, a=0, b=1):
    ts = unit(tau * r, a, b)
    c = (vunc(rect) * (r * ts)).T

    a = np.full(np.shape(c), ants[i].T[j])

    return c + a


def circ_ij(i, j, r, a=0., b=1.):
    ts = unit(tau * r, 1 - a, 1 - b)
    # print(exp2ni(ts))
    return complex(i, j) + r * ts


def diff(ants, i, j):
    return ants[i] - ants[j]


def line_ex(ants):
    """
    Draws a line between noses <3
    """

    # diff tests
    n = 150
    diff_vec = parse_diff("nose", 0, "nose", 1, ants)[0]
    # print("F", ns_01)
    lerp_mat = uint(n).T @ diff_vec.reshape((1, 2))
    # print(wn2bn01)
    base_pnt = coords("nose", 0, ants)[0]
    # print(base)
    pnts = np.apply_along_axis(lambda x: base_pnt - x, 1, lerp_mat)
    # print(ns_)
    noses = rovnd * pnts.T
    # print(wn2bn)

    return noses


def smooth(df, roll=2):
    dh = df.rolling(roll, center=True).mean()

    return dh


def angle(u, v):
    uu = np.linalg.norm(u)
    vv = np.linalg.norm(v)

    # Handle potential division by zero if a vector is a zero vector
    if uu == 0 or vv == 0:
        return 0.0

    ang = np.cross(u, v) / (uu * vv)
    ang = np.clip(ang, -1.0, 1.0)
    ang = np.arcsin(ang)
    print(ang)
    return ang


# displays a frame and some test lines, circles
def main():
    # mouse id
    m_id = 73

    # file for annotations, annotations as points
    video, ants = files(m_id)

    # lines
    a0 = draw_n_frames(video, ants, 0, n=10)
    a = a0

    # diff tests

    # white nose -> black nose
    wn2bn = line_ex(ants)
    a[wn2bn[1], wn2bn[0]] = [0, 0, 255]

    # a red circle!
    circ = circ_ij(300, 300, 10)
    xirc = rovnd * (vunc(rect) * circ.T)
    a[xirc[1], xirc[0]] = [255, 0, 0]

    # another green circle!
    dirc = circ_ij(400, 400, 20, 0, 0.3)
    yirc = rovnd * (vunc(rect) * dirc.T)
    a[yirc[1], yirc[0]] = [0, 255, 0]

    # a third blue circle!
    eirc = circ_ij(300, 400, 30, 0.25, -0.125)
    zirc = rovnd * (vunc(rect) * eirc.T)
    a[zirc[1], zirc[0]] = [0, 0, 255]

    plt.imshow(a)
    plt.show()

    cv2.destroyAllWindows()


# tests frame viewing control
# TODO: fix
def nain():  # mouse id

    col = lambda *x: np.array(x)
    swapaxes = lambda axis, bxis: func(lambda y: y.swapaxes(*x))
    moveaxis = lambda axis, bxis: func(lambda y: np.moveaxis(y, *x))
    ints = func(lambda x: np.vectorize(round)(x))
    #round = func(np.vectorize(int))
    eucl = func(lambda z: col(z.real, z.imag))

    m_id = 73

    log_debug["Loading!"]

    # file for annotations, annotations as points
    vid_path, vid_ants = files(m_id)
    vid_capt = get_video(vid_path)

    # figure setup
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('')
    ax.set_xticks([])
    ax.set_yticks([])

    success, frame = vid_capt.read()
    if not success:
        log_debug["Failed"]
        quit()

    plot = ax.imshow(frame)

    n = 0

    def on_key_press(event):
        nonlocal n

        def draw():
            vid_capt.set(cv2.CAP_PROP_POS_FRAMES, n)

            success, frame = vid_capt.read()
            if not success:
                log_debug["Failed"]
                quit()

            # mice x (X, Y) x #bodyparts x 3
            ants = ints * vid_ants[n]
            # each is #bodyparts x (X, Y) x 3
            ants_m1, ants_m2 = moveaxis(2, 1) * ants
            # text offset
            txts = []
            offset = np.array([10, 10])
            for i, ant in enumerate(ants_m1):
                circ = ints * eucl * circ_ij(*ant, 5)
                frame[*ant[R]] = RGB.R
                frame[*circ[R]] = RGB.R
                txts.append(plt.text(*ant + offset, bp_ids[i], color='r', fontsize=5))

            for i, ant in enumerate(ants_m2):
                circ = ints * eucl * circ_ij(*ant, 5)
                frame[*ant[R]] = RGB.B
                frame[*circ[R]] = RGB.B
                txts.append(plt.text(*ant + offset, bp_ids[i], color='b', fontsize=5))

            txts.append(plt.text(5, 30, f"{n}", color="r", fontsize=10))

            plot.set_array(frame)
            fig.canvas.draw()

            for txt in txts:
                txt.remove()

        if event.key == 'd':
            n += 1
            draw()
        elif event.key == 'a':
            n -= 1
            draw()

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()
    vid_capt.release()


# plots N1-N2, and d/dt(N1-N2) for multiple t-intervals
def oain():
    m_id = 73
    video, ants = files(m_id)

    diff_vec = parse_diff("nose", 0, "nose", 1, ants)

    d = lambda DF: (DF - DF.shift(-1))[:-1]

    #df = pd.DataFrame(diff_vec)

    f = pd.DataFrame(coords("tail", 1, ants))
    f.to_csv(f"C://Users/danie/code/work/extraction/out/{m_id}/calms21_{m_id}_wn2bn_diff0.csv", header=False,
             index=False)

    i = 4
    j = 20
    fi = f.rolling(i).mean().shift(-i // 2)
    fj = f.rolling(j).mean().shift(-j // 2)

    df = d(f)
    df_i = df.rolling(i).mean()
    df_j = df.rolling(j).mean()

    d_fi = d(fi)
    d_fi_i = d_fi.rolling(i).mean()
    d_fi_j = d_fi.rolling(j).mean()

    d_fj = d(fj)
    d_fj_i = d_fj.rolling(i).mean()
    d_fj_j = d_fj.rolling(j).mean()

    basic_plot(f[0], label=f"x")
    basic_plot(fi[0], label=f"x~{i}")
    basic_plot(fj[0], label=f"x~{j}")

    basic_plot(df[0], label=f"dx")
    basic_plot(df_i[0], label=f"dx~{i}")
    basic_plot(df_j[0], label=f"dx~{j}")
    # equivalent to above!
    #basic_plot(d_fi[0], label=f"d(x~{i})")
    basic_plot(d_fj[0], label=f"d(x~{j})")

    basic_plot(d_fi_i[0], label=f"d(x~{i})~{i}")
    basic_plot(d_fj_i[0], label=f"d(x~{j})~{i}")
    #basic_plot(d_fi_j[0], label=f"d(x~{i})~{j}")
    #basic_plot(d_fj_j[0], label=f"d(x~{j})~{j}")

    plt.legend()
    plt.show()


# plots N1 position, smoothed
def pain():
    # mouse id
    m_id = 73

    # file for annotations, annotations as points
    video, ants = files(m_id)

    nose = coords("nose", 0, ants)

    def pt(x):
        plt.plot(range(len(x)), x)

    #plt.imshow(a)
    nx = pd.DataFrame(nose[:, 0])
    pt(nx)
    pt(smooth(nx, 3))
    pt(smooth(nx, 5))
    plt.show()


# draws lines to EL/R
def qain():
    # mouse id
    m_id = 73

    # file for annotations, annotations as points
    video, ants = files(m_id)

    a = get_frame(0, video)

    dl = parse_diff("nose", 0, "earl", 0, ants)[0]
    dr = parse_diff("nose", 0, "earr", 0, ants)[0]

    erp = uint(100).T
    elrp = erp @ dl.reshape((1, 2))
    errq = erp @ dr.reshape((1, 2))

    base_pnt = coords("nose", 0, ants)[0]

    pnts = np.apply_along_axis(lambda x: base_pnt - x, 1, elrp)
    qnts = np.apply_along_axis(lambda x: base_pnt - x, 1, errq)
    # print(ns_)
    left = (rovnd * pnts).T
    rite = (rovnd * qnts).T

    a[left[1], left[0]] = [0, 0, 255]
    a[rite[1], rite[0]] = [255, 0, 0]

    plt.imshow(a)
    plt.show()


# draws lines to ears and the angle between
def rain():
    # mouse id
    m_id = 73

    # file for annotations, annotations as points
    video, ants = files(m_id)

    a = get_frame(0, video)

    dl = parse_diff("nose", 0, "earl", 0, ants)[0]
    dr = parse_diff("nose", 0, "earr", 0, ants)[0]

    # "interp(olation)"
    ntrp = uint(100).T
    lerp = ntrp @ dl.reshape((1, 2))
    rerp = ntrp @ dr.reshape((1, 2))

    base_pnt = coords("nose", 0, ants)[0]

    pnts = np.apply_along_axis(lambda x: base_pnt - x, 1, lerp)
    qnts = np.apply_along_axis(lambda x: base_pnt - x, 1, rerp)
    # print(ns_)

    left = (rovnd * pnts).T
    rite = (rovnd * qnts).T
    a[left[1], left[0]] = [0, 0, 255]
    a[rite[1], rite[0]] = [255, 0, 0]

    # find angle between ears, from the left

    # (but CalMS videos are flipped so l/r look switched
    u = np.angle(comp(*dl)) / (2 * np.pi)
    v = angle(dl, dr) / (2 * np.pi)

    # use u + v because the vector used for u (dl) is first in v.
    # cross product is anticommutative, use u - v if drawing reversed
    rnts = 20 * unit(100, u, u + v).T
    rnts = np.apply_along_axis(rect, 0, rnts).reshape((2, 100)).T
    rnts = np.apply_along_axis(lambda x: base_pnt - x, 0, rnts.T)
    angl = (rovnd * rnts)
    a[angl[1], angl[0]] = [255, 0, 255]

    plt.imshow(a)
    plt.show()


def sain():  # mouse id

    col = lambda *x: np.array(x)
    trns = lambda *x: func(lambda y: y.swapaxes(*x))
    move = lambda *x: func(lambda y: np.moveaxis(y, *x))
    ints = func(lambda x: np.vectorize(round)(x))
    #round = func(np.vectorize(int))
    eucl = func(lambda z: col(z.real, z.imag))

    m_id = 73

    log_debug["Loading!"]

    # file for annotations, annotations as points
    video, ants = files(m_id)
    vid = get_video(video)

    figure, axes = plt.subplots()
    figure.canvas.manager.set_window_title('')

    # Hide the axes ticks and labels for a cleaner look
    axes.set_xticks([])
    axes.set_yticks([])

    # --- 3. Initialize the Plot ---
    # Read the first frame to get the dimensions and initialize the plot.
    success, frame = vid.read()
    if not success:
        log_debug["Failed"]
        quit()

    im = axes.imshow(frame)

    n = 1600
    txts = []

    def targets(frame, points, labels, offset, color, radius, text=False):
        nonlocal txts

        for i, point in enumerate(points):
            circ = ints * eucl * circ_ij(*point, radius)
            frame[*point[R]] = color
            frame[*circ[R]] = color
            if text:
                txts += [plt.text(*point + offset, labels[i], color=(*color / 255,), fontsize=8)]

        return frame

    def on_key_press(event: Event):
        nonlocal n

        def draw():
            nonlocal n, txts
            vid.set(cv2.CAP_PROP_POS_FRAMES, n)

            success, frame = vid.read()
            if not success:
                log_debug["Failed"]
                quit()

            # text offset
            txts = []
            offset = np.array([10, 10])

            # given annotations
            # mice x (X, Y) x #bodyparts x 3
            antn = ints * ants[n]
            # each is #bodyparts x (X, Y) x 3
            ant, bnt = move(2, 1) * antn
            frame = targets(frame, ant, bp_ids, offset, RGB.R, 2, True)
            frame = targets(frame, bnt, bp_ids, offset, RGB.B, 2, True)

            # smoothed annotation
            mean = gaussian_filter1d(ants, 1.5, axis=0, mode='reflect')
            ant, bnt = move(2, 1) * ints * mean[n]
            frame = targets(frame, ant, bp_ids, offset, RGB.R + 1 / 2 * RGB.G, 5)
            frame = targets(frame, bnt, bp_ids, offset, RGB.B + 1 / 2 * RGB.G, 5)

            # top left info
            txts.append(plt.text(5, 30, f"{n}", color="r", fontsize=14))

            im.set_array(frame)
            figure.canvas.draw()

            for txt in txts:
                txt.remove()

        if event.key == 'd':
            n += 1
        elif event.key == 'a':
            n -= 1
        elif event.key == 'D':
            n += 50
        elif event.key == 'A':
            n -= 50
        else:
            return

        draw()

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()
    vid.release()


def tain():  # mouse id

    col = lambda *x: np.array(x)
    trns = lambda *x: func(lambda y: y.swapaxes(*x))
    move = lambda *x: func(lambda y: np.moveaxis(y, *x))
    ints = func(lambda x: np.vectorize(round)(x))
    #round = func(np.vectorize(int))
    eucl = func(lambda z: col(z.real, z.imag))
    d = lambda arr: arr[1:] - arr[:-1]

    m_id = 73

    log_debug["Loading!"]

    # file for annotations, annotations as points
    video, ants = files(m_id)
    nose = coords("nose", 1, ants)
    nose_norm = np.apply_along_axis(np.linalg.norm, 1, nose)
    d_nose_norm = d(nose_norm)
    dd_nose_norm = d(d_nose_norm)
    ddd_nose_norm = d(dd_nose_norm)
    dddd_nose_norm = d(ddd_nose_norm)
    diff = parse_diff("nose", 1, "tail", 1, ants)
    diff_norm = np.apply_along_axis(np.linalg.norm, 1, diff)
    d_diff_norm = d(diff_norm)
    dd_diff_norm = d(d_diff_norm)
    ddd_diff_norm = d(dd_diff_norm)
    dddd_diff_norm = d(ddd_diff_norm)

    vid = get_video(video)

    figure, axes = plt.subplots()
    figure.canvas.manager.set_window_title('')

    # Hide the axes ticks and labels for a cleaner look
    axes.set_xticks([])
    axes.set_yticks([])

    # --- 3. Initialize the Plot ---
    # Read the first frame to get the dimensions and initialize the plot.
    success, frame = vid.read()
    if not success:
        log_debug["Failed"]
        quit()

    #im = axes.imshow(frame)

    n = 0
    txts = []

    def targets(frame, points, labels, offset, color, radius, text=False):
        nonlocal txts

        for i, point in enumerate(points):
            circ = ints * eucl * circ_ij(*point, radius)
            frame[*point[R]] = color
            frame[*circ[R]] = color
            if text:
                txts += [plt.text(*point + offset, labels[i], color=(*color / 255,), fontsize=8)]

        return frame

    def on_key_press(event: Event):
        nonlocal n

        def draw():
            nonlocal n, txts
            vid.set(cv2.CAP_PROP_POS_FRAMES, n)

            success, frame = vid.read()
            if not success:
                log_debug["Failed"]
                quit()

            # text offset
            txts = []
            offset = np.array([10, 10])

            # given annotations
            # mice x (X, Y) x #bodyparts x 3
            antn = ints * ants[n]
            # each is #bodyparts x (X, Y) x 3
            ant, bnt = move(2, 1) * antn
            frame = targets(frame, ant, bp_ids, offset, RGB.R, 2, True)
            frame = targets(frame, bnt, bp_ids, offset, RGB.B, 2, True)

            # smoothed annotation
            #mean = gaussian_filter1d(ants, 1.5, axis=0, mode='reflect')
            #ant, bnt = move(2, 1) * ints * mean[n]
            #frame = targets(frame, ant, bp_ids, offset, RGB.R + 1 / 2 * RGB.G, 5)
            #frame = targets(frame, bnt, bp_ids, offset, RGB.B + 1 / 2 * RGB.G, 5)

            # top left info
            print(diff_norm)
            infos = [f"{n}", f"HT diff_norm: {diff_norm[n]}", f"HT' diff_norm: {d_diff_norm[n]}"]
            info = "\n".join(infos)
            txts.append(plt.text(5, 30, info, color="r", fontsize=14))

            #im.set_array(frame)
            figure.canvas.draw()

            for txt in txts:
                txt.remove()

        if event.key == 'd':
            n += 1
            draw()
        elif event.key == 'a':
            n -= 1
            draw()

    #figure.canvas.mpl_connect('key_press_event', on_key_press)

    x = len(dddd_diff_norm)
    X = range(x)
    #plt.plot(X, diff_norm[:x])
    #plt.plot(X, d_diff_norm[:x])
    #plt.plot(X, dd_diff_norm[:x])
    #plt.plot(X, ddd_diff_norm[:x])
    #plt.plot(X, dddd_diff_norm[:x])
    #plt.plot(X, nose_norm[:x])
    plt.plot(X, d_nose_norm[:x])
    plt.plot(X, dd_nose_norm[:x])
    plt.plot(X, ddd_nose_norm[:x])
    plt.plot(X, dddd_nose_norm[:x])

    plt.show()
    vid.release()


if __name__ == '__main__':

    fs = {
        "m": main,  # annotations, circles, lines
        "n": nain,  # annotations as labeled targets
        "o": oain,  # tail (x) position, smoothing
        "p": pain,  # trying easy smoothing
        "q": qain,  # point difference extraction
        "r": rain,  # angle extraction
        "s": sain,  # annotations + smoothing as labeled targets
        "t": tain   # annotations + diffs, diffs' to find errors
    }

    if len(sys.argv) > 1:
        f = sys.argv[1]
        fs[f]()
        quit()

    tain()

    '''
    TODO:
        - figure out behaviors
            investigate (search if stretching!)
            attack
            sniff nose/side/back
            rear
            walk/run, chase
                - moving if dB > 0
                - chasing if dBi, dBj > 0
                  and 
       
        - parse data:
            SINGLE
            position
            distances
            angles
            facing direction
            body direction
            velocities
            d(theta) ?
            
            SOCIAL
            basics
                ^ = angle

                N = nose, M = neck, T = tail
                EX = L/R ear, HX = L/R hip
                F = facing vectors;
                B = body vector;
                  = |Ni Mi| + |Mi Ti|
                dX = dX/dt

                Ni ~ Nj
                    facing each other
                    - B, F doesnt matter
                Ni ~ Tj 
                    sniffing rear
                    - B, F dont matter
                Ni ~ Hj
                    sniffing side
                    - B, F dont matter
            useful
                Bi ^ Fi
                    - turning if >> 0
                
            

        - put data into csv
    '''

    #quit()
