import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as DF

# GLOBAL VARIABLES  
# num frames to average values over
frame_window = 5
partslist_0 = ["mouse_0", "nose_0", "left_0", "right_0", "tail_0"]
partslist_1 = ["mouse_1", "nose_1", "left_1", "right_1", "tail_1"]

def xy(data):
    return data.loc[:, "x"], data.loc[:, "y"]

def x(data):
    return xy(data)[0]

def y(data):
    return xy(data)[1]

def apply(f, *x):
    return [f(xi) for xi in x]

def shift(df: DF, m = 1, fill=0):
    return df.shift(m, fill_value=fill)  # i just prefer passing df

def mean(df: DF, *labels):
    return 1 / len(labels) * sum([df.loc[:, l] for l in labels])

# averages window frames
def n_mean(df, n = frame_window):
    # f = frames(df)
    # df.drop("frame_number", axis = 1, inplace=True)
    dfs = [shift(df, m = -i) for i in range(n)]

    dg = sum(dfs)
    dh = pd.concat([1/n * dg], axis = 1)
    # print(f, dh)
    return dh

def frames(df):
    return df.index

def pos(df: DF):
    f = frames(df)
    x = mean(df, "x1", "x2")
    y = mean(df, "y1", "y2")

    ret = pd.concat([x, y], axis=1)
    # print(ret)
    ret = ret.rename(columns={0: "x", 1: "y"})
    return ret

def labels_filled(data, *labels):
    w = lambda l: data.where(data["instance_name"] == l).ffill()
    return [w(l) for l in labels]

def labels_dropped(data, *labels):
    w = lambda l: data.where(data["instance_name"] == l).dropna(how="any")
    return [w(l) for l in labels]

def mag(data):
    x, y = xy(data)
    return (x**2 + y**2)**(1/2)

def plot(df):
    plt.plot(df.index, df)

def main():
    # csv_ = "CaMKII 1-8 M R 2-8-21-Phase 3_tracking-240906.csv"
    # csv_ = "cb_tracking.csv"
    csv_ = "0.35-X-total-unkeyed_Cd1-Social-Test-HFlip-crop_coco_dataset_cb_Cd1-Social-Test-crop_1000_iters_mask_rcnn_tracking_results_with_segmenation.csv"
    data = pd.read_csv(csv_)
    data = data.set_index("frame_number")
    
    dn =  lambda df, m = frame_window: df - n_mean(df, m)

    # labels_filled,
    # labels_dropped
    _f = labels_filled
    parts_0 = _f(data, *partslist_0)
    parts_1 = _f(data, *partslist_1)

    mouse_0, nose_0, left_0, right_0, tail_0 = parts_0
    mouse_1, nose_1, left_1, right_1, tail_1 = parts_1

    pos_0 = apply(pos, *parts_0)
    pos_1 = apply(pos, *parts_1)
    pos_0_m, pos_0_n, pos_0_l, pos_0_r, pos_0_t = pos_0
    pos_1_m, pos_1_n, pos_1_l, pos_1_r, pos_1_t = pos_1
    
    vel_0 = apply(dn, *pos_0)
    vel_1 = apply(dn, *pos_1)
    vel_0_m, vel_0_n, vel_0_l, vel_0_r, vel_0_t = vel_0
    vel_1_m, vel_1_n, vel_1_l, vel_1_r, vel_1_t = vel_1
    
    acc_0 = apply(dn, *vel_0)
    acc_1 = apply(dn, *vel_1)
    acc_0_m, acc_0_n, acc_0_l, acc_0_r, acc_0_t = acc_0
    acc_1_m, acc_1_n, acc_1_l, acc_1_r, acc_1_t = acc_1

    diff_n = pos_0_n - pos_1_n

    # vel_0, vel_1 = d1(pos_0), d1(pos_1)
    # acc_0, acc_1 = d1(vel_0), d1(vel_1)

    # mos_m0n, pos_1n = n_mean(pos_0, n), n_mean(pos_1, n)
 
    # print(pos_0, pos_1)
    # print(vel_0, vel_0)
    # print(acc_0, acc_1)

    # len(acc_0.iloc[:, 0])
    # len(acc_01.iloc[:, 0])
    r0 = 0
    r1 = 500
    rn = range(r0, r1)
    rs = slice(r0, r1)
    # plt.plot(rn, x(pos_0n)[rs])
    # plt.plot(rn, y(pos_0n)[rs])
    # plt.plot(rn, y(pos_1n)[rs])

    # plt.plot(rn, y(dn(pos_0n))[rs])
    # plt.plot(rn, y(dn(pos_1n))[rs])

    # print(pos_0_m)
    # print(vel_0_m)
    # print(acc_0_m)

    # plt.plot(frames(mouse_0)[rs], n_mean(pos_0_m)[rs])
    # plt.plot(frames(mouse_0)[rs], x(n_mean(vel_0_m))[rs])
    # plt.plot(frames(mouse_0)[rs], x(n_mean(acc_0_m))[rs])

    print(diff_n)
    plt.plot(diff_n.index, mag(n_mean(diff_n)))

    # plt.plot(rn, n_mean(mag(pos_0_l))[rs])
    # plt.plot(rn, n_mean(mag(pos_0_n))[rs])
    # plt.plot(rn, n_mean(mag(pos_0_r))[rs])
    # plt.plot(rn, n_mean(mag(pos_0_t))[rs])
    # plt.legend()

    # print(x(n_mean(vel_0_m)))

    plt.show()


if __name__ == "__main__":
    main()
