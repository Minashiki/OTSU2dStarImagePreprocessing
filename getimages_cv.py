import numpy as np
import cv2
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from show_fits import show_images_grid, zscale_asinh

# =========================
# 1. FITS 读入 & 基础处理
# =========================

def read_fits_as_float32(path, crop=None):
    """
    读入 FITS 图像，转成 float32，并把 NaN 用中位数填掉。
    这一部分论文没规定，是 IO 和数值稳定性处理。
    """
    with fits.open(path) as hdul:
        # data = hdul[ext].data.astype(np.float32)
        data = fits.getdata(path).astype(np.uint8)
        if crop != None:
            h, w = data.shape[:2]
            ch, cw = crop
            y1 = max(h // 2 - ch // 2, 0)
            y2 = min(h // 2 + ch // 2, h)
            x1 = max(w // 2 - cw // 2, 0)
            x2 = min(w // 2 + cw // 2, w)
            data = data[y1:y2, x1:x2]
    # 用整体中位数替换 NaN
    med = np.nanmedian(data)
    data = np.nan_to_num(data, nan=med, posinf=med, neginf=med)
    print(f"图像尺寸: {data.shape}, 最小值={data.min():.1f}, 最大值={data.max():.1f}")
    return data, data.max()


# =========================
# 2. 2D 中值滤波（滑窗）
# =========================

def median_filter_2d(image, radius):
    """
    纯 NumPy 实现的 2D 滑窗中值滤波，对应论文里的 mf_R(·)。

    为了避免 SciPy，这里用 pad + stride 的方式实现。
    边界条件用 reflect（和 PDF 里的“滑窗中值滤波”思想一致）。
    """
    k = 2 * radius + 1           # 窗口边长
    if k <= 1:
        return image.copy()

    img = image.astype(np.float32)
    pad = radius
    # 反射边界
    padded = np.pad(img, pad_width=pad, mode="reflect")

    H, W = img.shape
    # 构造滑窗视图 (H, W, k, k)
    shape = (H, W, k, k)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1],
    )
    patches = np.lib.stride_tricks.as_strided(
        padded, shape=shape, strides=strides
    )
    # 对 (k, k) 这两个维度做中值
    med = np.median(patches, axis=(2, 3))
    return med.astype(np.float32)
    # from scipy.ndimage import median_filter
    # ksize = 2 * radius + 1
    #
    # # SciPy 的 median_filter 是 C 实现的，也不会爆内存
    # # 还能处理超大窗口（100, 200, 400 都没问题）
    # filtered = median_filter(image, size=ksize, mode='reflect')
    # return filtered


# =========================
# 3. 局部标准差（用 OpenCV 的 box filter）
# =========================

def local_std_cv(image, ksize=3):
    """
    对应论文里的 D_lambda = sd_9(S~_lambda)：
    用 3x3 滑窗计算局部标准差。
    这里用 OpenCV 的均值滤波（box filter）来实现。
    """
    img = image.astype(np.float32)
    k = (ksize, ksize)
    # E[X]
    mean = cv2.blur(img, ksize=k, borderType=cv2.BORDER_REFLECT_101)
    # E[X^2]
    mean_sq = cv2.blur(img * img, ksize=k, borderType=cv2.BORDER_REFLECT_101)
    var = np.clip(mean_sq - mean * mean, a_min=0.0, a_max=None)
    return np.sqrt(var, dtype=np.float32)


# =========================
# 4. 多尺度窗口半径构建（R_{λj}）
# =========================

def build_window_radii(beam_fwhm_pix, X_fwhm_pix, fW=np.sqrt(2.0)):
    """
    按论文构建窗口半径：
        R1 = 2 * O_lambda
        Rj = fW * R_{j-1}
        RN ≈ 4 * X_lambda

    这里所有单位都要求你已经换算成像素：
        beam_fwhm_pix = O_lambda (PSF FWHM in pixels)
        X_fwhm_pix    = X_lambda (最大感兴趣结构 FWHM in pixels)
    """
    R1 = 2.0 * float(beam_fwhm_pix)
    R_target = 4.0 * float(X_fwhm_pix)

    radii = []
    r = R1
    while r <= R_target * 1.01:  # 稍微放宽一点
        radii.append(int(round(r)))
        r *= fW

    radii = sorted(set(max(1, int(rr)) for rr in radii))
    return radii


# =========================
# 5. 多尺度中值滤波 + 逐点最小（Eq. (2), (6)）
# =========================

def multiscale_median_min(image, radii):
    """
    对应：
        Eq. (2): B_hat = min_j mf_{R_j}(S_prev)
        Eq. (6): F_hat = min_j mf_{R_j}(D_lambda)

    对同一图像用不同半径的滑窗中值滤波，然后逐点取最小。
    """
    img = image.astype(np.float32)
    filtered_list = []

    for r in radii:
        filtered = median_filter_2d(img, radius=r)
        filtered_list.append(filtered)

    stacked = np.stack(filtered_list, axis=0)  # (N_radii, H, W)
    return np.min(stacked, axis=0)


# =========================
# 6. 背景估计（Eqs. 2–5）
# =========================

def estimate_background_getimages(
    image,
    beam_fwhm_pix,
    X_fwhm_pix,
    fW=np.sqrt(2.0),
    M=30,
):
    """
    严格按论文的 getimages 背景推导部分：

    - 多尺度中值滤波 + min -> B_hat^i
    - 再用最大窗口平滑一次取 min -> B_i
    - B~ = sum_i B_i
    - S~ = I - B~

    参数:
        image         : 2D numpy array, 原始图像 I_lambda
        beam_fwhm_pix : O_lambda (PSF FWHM in pixels)
        X_fwhm_pix    : X_lambda (max structure FWHM in pixels)
        fW            : 尺度因子，论文默认 sqrt(2)
        M             : 迭代次数，论文建议 ~20–30

    返回:
        B_tilde : 估计的背景图 B~_lambda
        S_tilde : 背景减除图 S~_lambda = I - B~
        radii   : 使用的窗口半径列表（给 flattening 复用）
    """
    I = image.astype(np.float32)
    radii = build_window_radii(beam_fwhm_pix, X_fwhm_pix, fW=fW)
    RN = radii[-1]

    B_accum = np.zeros_like(I, dtype=np.float32)
    S_prev = I.copy()  # S~^0 = I

    for i in range(M):
        # Eq. (2)
        B_hat = multiscale_median_min(S_prev, radii)
        # Eq. (3) : 再用最大窗口做一次中值滤波并取 min
        B_hat_smooth = median_filter_2d(B_hat, radius=RN)
        B_i = np.minimum(B_hat, B_hat_smooth)

        B_accum += B_i
        S_prev = I - B_accum

    B_tilde = B_accum
    S_tilde = I - B_tilde
    return B_tilde, S_tilde, radii


# =========================
# 7. 图像展平（flattening，Eqs. 6–7）
# =========================

def flatten_image_getimages(S_tilde, radii):
    """
    按论文 Sect. 2.3, Eqs. (6)-(7) 展平：

    1) D = sd_9(S~) —— 3x3 局部标准差
    2) F_hat = min_j mf_{R_j}(D)
    3) F~ = min{ F_hat, mf_{R_N}(F_hat), mf_{R_N}(mf_{R_N}(F_hat)) }
    4) I^D = S~ / F~
    """
    S = S_tilde.astype(np.float32)
    RN = radii[-1]

    # D_lambda = sd_9(S~_lambda)
    D = local_std_cv(S, ksize=3)

    # Eq. (6): F_hat
    F_hat = multiscale_median_min(D, radii)

    # Eq. (7): 再用最大窗口做两次中值 + min
    F1 = median_filter_2d(F_hat, radius=RN)
    F2 = median_filter_2d(F1, radius=RN)
    F_tilde = np.minimum.reduce([F_hat, F1, F2])

    # 数值安全：避免除零（这一点论文没说，是实现细节）
    positive = F_tilde[F_tilde > 0]
    if positive.size > 0:
        eps = np.median(positive) * 1e-3
    else:
        eps = 1e-6
    F_safe = np.where(F_tilde > eps, F_tilde, eps)

    I_D = S / F_safe
    return F_tilde, I_D


# =========================
# 8. 一站式管线封装
# =========================

def getimages_pipeline_cv(
    image,
    beam_fwhm_pix,
    X_fwhm_pix,
    fW=np.sqrt(2.0),
    M=30,
):
    """
    完整 getimages 流程（无 SciPy）：
      1) 背景推导：B~, S~
      2) 展平：F~, I^D
    """
    B_tilde, S_tilde, radii = estimate_background_getimages(
        image,
        beam_fwhm_pix=beam_fwhm_pix,
        X_fwhm_pix=X_fwhm_pix,
        fW=fW,
        M=M,
    )

    F_tilde, I_D = flatten_image_getimages(S_tilde, radii)

    return {
        "B_tilde": B_tilde,
        "S_tilde": S_tilde,
        "F_tilde": F_tilde,
        "I_D": I_D,
        "radii": radii,
    }


# =========================
# 9. 示例：对一幅 FITS 星图运行
# =========================

def demo_on_fits(
    fits_path,
    beam_fwhm_pix,
    X_fwhm_pix,
    ext=0,
    M=30,
    crop=None
):
    """
    演示如何在真实 FITS 星图上跑 getimages。

    你只需要改：
      - fits_path      : 你的星图路径
      - beam_fwhm_pix  : PSF FWHM（像素）
      - X_fwhm_pix     : 最大感兴趣结构 FWHM（像素）
    """
    img, data_max = read_fits_as_float32(fits_path, crop=crop)

    result = getimages_pipeline_cv(
        image=img,
        beam_fwhm_pix=beam_fwhm_pix,
        X_fwhm_pix=X_fwhm_pix,
        M=M,
    )

    B_tilde = result["B_tilde"]
    S_tilde = result["S_tilde"]
    F_tilde = result["F_tilde"]
    I_D = result["I_D"]

    # # 下面只是简单把结果归一化后存成 PNG/TIFF，便于肉眼查看
    # def to_8bit(x, clip_percent=0.5):
    #     x = x.astype(np.float32)
    #     lo, hi = np.percentile(x, [clip_percent, 100 - clip_percent])
    #     x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    #     return (x * 255).astype(np.uint8)
    #
    # cv2.imwrite("output_raw.png",  to_8bit(img))
    # cv2.imwrite("output_bg.png",   to_8bit(B_tilde))
    # cv2.imwrite("output_src.png",  to_8bit(S_tilde))
    # cv2.imwrite("output_flat.png", to_8bit(I_D))

    # show_images_grid(
    #     [img, B_tilde, S_tilde, I_D],
    #     ["raw", "B~", "S~", "I^D"],
    #     ncols=2,
    #     norm=zscale_asinh(img)
    # )
    print("保存了四张图：")
    print("  - output_raw.png   原始星图（简单归一化）")
    print("  - output_bg.png    估计的背景 B~")
    print("  - output_src.png   背景减除后的 S~")
    print("  - output_flat.png  展平后的 I^D（用于源检测）")
    return S_tilde, data_max


if __name__ == "__main__":
    # 这里你需要根据自己的望远镜 / 相机参数填入像素尺度
    # 下面这些数字只是示例，不是论文里的固定值！
    example_fits_path = "20240306204703518_059051_01_L/20240306204801957_6002.fits"

    # TODO: 换成你自己的参数（像素单位）：
    beam_fwhm_pix = 3.1   # 你这台设备测出来的 PSF 宽度
    X_fwhm_pix = 10.0  # 例如感兴趣结构最大 FWHM ≈ 20 像素
    M_iter        = 30

    demo_on_fits(
        fits_path=example_fits_path,
        beam_fwhm_pix=beam_fwhm_pix,
        X_fwhm_pix=X_fwhm_pix,
        ext=0,
        M=M_iter,
        crop=(256, 256)
    )
