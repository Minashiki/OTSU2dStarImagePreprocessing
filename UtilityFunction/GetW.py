import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from getimages_cv import demo_on_fits
from UtilityFunction.Compute2dHist import *
FITS_PATH = "20240306204703518_059051_01_L/20240306204801957_6002.fits"

def compute_prefix_sum(p):
    """
    构建二维前缀和 P，使得任意区域求和 O(1)
    P[x,y] = sum(p[0:x, 0:y])
    """
    return p.cumsum(axis=0).cumsum(axis=1)


def get_W0(P, s, t):
    """
    背景类概率 W0(s,t)
    W0 = sum(p[0:s, 0:t])
    """
    return P[s, t]


def get_W1(P, L, s, t):
    """
    目标类概率 W1(s,t)
    sum(p[s+1:L-1, t+1:L-1])
    利用前缀和矩阵 P 快速计算
    """
    total = P[L-1, L-1]
    part1 = P[s, L-1]
    # print("part1", part1)
    part2 = P[L-1, t]
    part3 = P[s, t]
    # print("part3", part3)
    return total - part1 - part2 + part3

if __name__ == '__main__':
    beam_fwhm_pix = 3.1
    X_fwhm_pix = 10.0
    M_iter = 30

    img, data_max = demo_on_fits(
        fits_path=FITS_PATH,
        beam_fwhm_pix=beam_fwhm_pix,
        X_fwhm_pix=X_fwhm_pix,
        ext=0,
        M=M_iter,
        crop=(512, 512)
    )
    p_ij = compute_2d_hist(img)
    # visualize_3d_hist(p_ij, save_path="C:\\Users\Administrator\Desktop\北理工\PreProcess")
    # 假设已经得到二维直方图 p_ij
    P = compute_prefix_sum(p_ij)

    s, t = 123, 87
    L = p_ij.shape[0]    # 而不是 data_max+1


    W0 = get_W0(P, s, t)
    W1 = get_W1(P, L, s, t)

    print("W0 =", W0)
    print("W1 =", W1)
    print("W0 + W1 =", W0 + W1)  # 接近 1（平原区假设忽略）
