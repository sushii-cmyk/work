import numpy as np

pall = lambda *xs: print("; ".join(xs))

def main():
    a = [0]
    A = np.array(a)
    B = np.array([a])

    print(f"{a=}, {A=}, {A.shape=}, {B=}, {B.shape=}")


if __name__ == '__main__':
    main()