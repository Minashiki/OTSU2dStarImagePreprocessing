import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from getimages_cv import *
from UtilityFunction.Compute2dHist import *
from show_fits import show_images_grid, zscale_asinh
from UtilityFunction.GetW import *
FITS_PATH = "20240306204703518_059051_01_L/20240306204801957_6002.fits"
def compute_weighted_prefix_sums(p):
    """
    构建三个前缀和：
    P   : p_ij 的前缀和
    Pi  : i * p_ij 的前缀和
    Pj  : j * p_ij 的前缀和
    """
    L = p.shape[0]
    I = np.arange(L).reshape(L, 1)  # i
    J = np.arange(L).reshape(1, L)  # j

    Pi = p * I
    Pj = p * J

    P = p.cumsum(axis=0).cumsum(axis=1)
    Pi = Pi.cumsum(axis=0).cumsum(axis=1)
    Pj = Pj.cumsum(axis=0).cumsum(axis=1)

    return P, Pi, Pj


def region_sum(P, x1, y1, x2, y2):
    """
    求二维区域和，区域为 [x1:x2, y1:y2]（含 x1,x2)
    利用前缀和公式，时间复杂度 O(1)
    """
    s = P[x2, y2]
    if x1 > 0: s -= P[x1-1, y2]
    if y1 > 0: s -= P[x2, y1-1]
    if x1 > 0 and y1 > 0: s += P[x1-1, y1-1]
    return s


def compute_mean_vectors(P, Pi, Pj, s, t, L):
    """
    计算：
        U0(s,t)
        U1(s,t)
        UT（只需一次）
    """
    # === 类概率 ===
    W0 = region_sum(P, 0, 0, s, t)
    W1 = region_sum(P, s+1, t+1, L-1, L-1)

    # === 背景类 U0 ===
    U0_i = region_sum(Pi, 0, 0, s, t) / W0
    U0_j = region_sum(Pj, 0, 0, s, t) / W0

    # === 目标类 U1 ===
    U1_i = region_sum(Pi, s+1, t+1, L-1, L-1) / W1
    U1_j = region_sum(Pj, s+1, t+1, L-1, L-1) / W1

    # === 总均值（可提前计算） ===
    total_i = region_sum(Pi, 0, 0, L-1, L-1)
    total_j = region_sum(Pj, 0, 0, L-1, L-1)
    UT_i = total_i
    UT_j = total_j

    return (U0_i, U0_j), (U1_i, U1_j), (UT_i, UT_j), W0, W1

def compute_weighted_prefix_sums(p):
    """
    和上一步一样：构建 P, Pi, Pj 前缀和
    """
    L = p.shape[0]
    I = np.arange(L).reshape(L, 1)  # i 方向
    J = np.arange(L).reshape(1, L)  # j 方向

    Pi = p * I
    Pj = p * J

    P  = p.cumsum(axis=0).cumsum(axis=1)
    Pi = Pi.cumsum(axis=0).cumsum(axis=1)
    Pj = Pj.cumsum(axis=0).cumsum(axis=1)

    # 总均值向量 U_T （注意：总概率 sum p_ij = 1）
    UT_i = Pi[-1, -1]  # sum i p_ij
    UT_j = Pj[-1, -1]  # sum j p_ij

    return P, Pi, Pj, UT_i, UT_j


def region_sum(P, x1, y1, x2, y2):
    """
    前面已经写过的区域求和函数，这里再贴一次方便你复制
    区域为 [x1:x2, y1:y2]（包含边界）
    """
    s = P[x2, y2]
    if x1 > 0: s -= P[x1-1, y2]
    if y1 > 0: s -= P[x2, y1-1]
    if x1 > 0 and y1 > 0: s += P[x1-1, y1-1]
    return s


def trace_SB_for_threshold(P, Pi, Pj, UT_i, UT_j, s, t):
    """
    给定阈值 (s,t)，计算 tr(S_B(s,t))
    P, Pi, Pj : 对应 p_ij, i p_ij, j p_ij 的前缀和
    UT_i, UT_j: 总均值向量 U_T
    """
    L = P.shape[0]

    # === 类概率 W0, W1 ===
    # 背景：左上角 [0..s, 0..t]
    W0 = region_sum(P, 0, 0, s, t)
    # 目标：右下角 [s+1..L-1, t+1..L-1]
    if s+1 <= L-1 and t+1 <= L-1:
        W1 = region_sum(P, s+1, t+1, L-1, L-1)
    else:
        W1 = 0.0

    # 避免除零或几乎没有像素的情况
    if W0 <= 1e-12 or W1 <= 1e-12:
        return 0.0

    # === 背景均值 U0 ===
    U0_i = region_sum(Pi, 0, 0, s, t) / W0
    U0_j = region_sum(Pj, 0, 0, s, t) / W0

    # === 目标均值 U1 ===
    U1_i = region_sum(Pi, s+1, t+1, L-1, L-1) / W1
    U1_j = region_sum(Pj, s+1, t+1, L-1, L-1) / W1

    # === 按论文公式计算 tr(S_B) ===
    d0_i = U0_i - UT_i
    d0_j = U0_j - UT_j
    d1_i = U1_i - UT_i
    d1_j = U1_j - UT_j
    # print(f"图像尺寸: {img.shape}, 最小值={img.min():.1f}, 最大值={img.max():.1f}", {img.dtype})
    tr_SB = W0 * (d0_i**2 + d0_j**2) + W1 * (d1_i**2 + d1_j**2)
    return tr_SB

def otsu_2d_find_thresholds(p):
    """
    输入：二维直方图 p_ij（概率归一化）
    输出：最优阈值 (s*, t*)
    """
    P, Pi, Pj, UT_i, UT_j = compute_weighted_prefix_sums(p)
    L = p.shape[0]

    best_s, best_t = 0, 0
    best_score = -1.0

    for s in range(L-1):
        for t in range(L-1):
            score = trace_SB_for_threshold(P, Pi, Pj, UT_i, UT_j, s, t)
            if score > best_score:
                best_score = score
                best_s, best_t = s, t

    return best_s, best_t, best_score

def apply_2d_otsu_threshold(img, s, t):
    """
    img：原图 (uint8)
    s, t：二维 otsu 阈值
    输出：二值图 mask (0/255)
    """
    # 计算邻域平均（3x3）
    kernel = np.ones((7, 7), np.float32) / 49
    mean_img = cv2.filter2D(img.astype(np.float32), -1, kernel)
    mean_img = np.clip(mean_img, 0, 255).astype(np.uint8)

    # 条件：g > s & m > t → 目标
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[(img > s) & (mean_img > t)] = 255
    return mask

if __name__ == '__main__':
    beam_fwhm_pix = 3.1
    X_fwhm_pix = 10.0
    M_iter = 30

    # img, data_max = demo_on_fits(
    #     fits_path=FITS_PATH,
    #     beam_fwhm_pix=beam_fwhm_pix,
    #     X_fwhm_pix=X_fwhm_pix,
    #     ext=0,
    #     M=M_iter,
    #     crop=(256,256)
    # )
    fits_path = "20240306204703518_059051_01_L/20240306204801957_6002.fits"
    img, data_max = read_fits_as_float32(fits_path, crop=(256, 256))
    print(f"图像尺寸: {img.shape}, 最小值={img.min():.1f}, 最大值={img.max():.1f}", {img.dtype})
    img_median = cv2.medianBlur(img, ksize=3)
    img_gaussian = cv2.GaussianBlur(img_median, ksize=(3, 3), sigmaX=1.0, sigmaY=1.0)
    img = img_gaussian.copy()
    print(f"图像尺寸: {img.shape}, 最小值={img.min():.1f}, 最大值={img.max():.1f}", {img.dtype})
    print(img)
    show_images_grid(
        [img],
        ["raw"],
        ncols=2,
        norm=zscale_asinh(img),
        # norm=None,
    )
    p_ij = compute_2d_hist(img)
    visualize_3d_hist(p_ij, save_path="C:\\Users\Administrator\Desktop\北理工\PreProcess")
    # 假设已经得到二维直方图 p_ij
    P = compute_prefix_sum(p_ij)

    s, t = 50, 30
    L = p_ij.shape[0]    # 而不是 data_max+1


    W0 = get_W0(P, s, t)
    W1 = get_W1(P, L, s, t)

    print("W0 =", W0)
    print("W1 =", W1)
    print("W0 + W1 =", W0 + W1)  # 接近 1（平原区假设忽略）

    # Step 2~4：搜索最佳阈值
    s_star, t_star, score = otsu_2d_find_thresholds(p_ij)
    print("最佳 阈值 s、t =", s_star, t_star)
    print("最大类间散度 =", score)

    # Step 5：二值化
    mask = apply_2d_otsu_threshold(img, s_star, t_star)
    show_images_grid(
        [img, mask],
        ["raw", "mask"],
        ncols=2,
        norm=zscale_asinh(img),
        # norm=None,
    )
    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("2D Otsu mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()