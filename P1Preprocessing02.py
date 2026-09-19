import os
import warnings
import cv2
import pandas as pd
import csv
from PIL import Image
import numpy as np
import hashlib
import struct
from UtilityFunction.ComputeMeanVector import *
from UtilityFunction.calculate_centroid import *
from UtilityFunction.CalculatePrecisionRecall import calculate_precision_recall, calculate_precision_recall_fits

'''FITS真实星图用的'''

FITS = True
image_path = "JPEGImages/Simu_0.779_-1.3433_-0.99016.jpg"
tag_img_path = "SegmentationClass/Simu_0.779_-1.3433_-0.99016.png"

FITS_PATH = "rst19\\rst19\\20260330163205413_9901.fits"
fits_num = "01"
star_num = str(int(fits_num))
TAG_FITS_PATH = "20240306204703518_059051_01_L/每张星图识别出的星点/starnum"+star_num+".csv"
fits_name = "note_fits" + fits_num
save_dir = "rst19\\" + fits_name

save_dir_centroids = save_dir + "/Centroid_Top"
TOP_NUM = 5000
PEAK_TRANSFORM_MIN = 7
PEAK_TRANSFORM_MAX = 30
S_INIT = 7
T_INIT = 30
os.makedirs(save_dir, exist_ok=True)
def load_fits_float32(fits_path, hdu_index=0, memmap=False):
    """
    读取 FITS 主图像为 float32，并给出稳健统计，捕获“可能截断”警告。
    """
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        hdul = fits.open(fits_path, memmap=memmap)
        try:
            data = hdul[hdu_index].data
            header = hdul[hdu_index].header
        finally:
            hdul.close()

    if data is None:
        raise ValueError(f"HDU[{hdu_index}] 没有图像数据")

    img = np.array(data, dtype=np.float32, copy=True)

    # 记录 warning（特别关注截断）
    warn_msgs = [str(w.message) for w in wlist]
    truncated = any("may have been truncated" in m for m in warn_msgs)

    # 稳健统计（分位数比 min/max 更有意义）
    p = np.percentile(img[np.isfinite(img)], [0.1, 1, 50, 99, 99.9])
    stats = {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "median": float(np.median(img)),
        "std": float(np.std(img)),
        "p0.1": float(p[0]),
        "p1": float(p[1]),
        "p50": float(p[2]),
        "p99": float(p[3]),
        "p99.9": float(p[4]),
        "truncated_warning": truncated,
        "warnings": warn_msgs,
    }
    return img, header, stats

def save_float_as_png_uint16(img_f32, save_path, lo_q=0.1, hi_q=99.9):
    finite = img_f32[np.isfinite(img_f32)]
    lo, hi = np.percentile(finite, [lo_q, hi_q])
    x = np.clip(img_f32, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    x16 = (x * 65535.0).astype(np.uint16)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(x16, mode="I;16").save(save_path)

def median_filter_3x3(image):
    """
    3×3中值滤波（OpenCV API实现）
    :param image: 输入图像（numpy数组，支持灰度图/彩色图）
    :return: 滤波后的图像
    """
    if image is None:
        raise ValueError("输入图像为空，请检查路径或图像完整性")
    # ksize=3：3×3滤波核，OpenCV自动处理边界（默认填充）
    return cv2.medianBlur(image, ksize=3)

def gaussian_filter_3x3(image, sigmaX=1.0):
    """
    3×3高斯滤波（OpenCV API实现）
    :param image: 输入图像（numpy数组，支持灰度图/彩色图）
    :param sigmaX: X方向标准差（控制模糊程度，默认1.0，>0时自动计算Y方向标准差）
    :return: 滤波后的图像
    """
    if image is None:
        raise ValueError("输入图像为空，请检查路径或图像完整性")
    # ksize=(3,3)：3×3滤波核，sigmaY默认=sigmaX
    return cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=sigmaX, sigmaY=sigmaX)

def compute_kernel(img):
    filtered_image = cv2.boxFilter(
        src=img,
        ddepth=-1,  # 输出图像深度与输入一致（-1表示自动匹配）
        ksize=(3, 3),
        normalize=True,
        borderType=cv2.BORDER_DEFAULT
    )
    return filtered_image

def test(img, f_img, s, j):
    H,W = img.shape
    result = np.zeros(shape=(H,W), dtype=np.uint8)
    for h in range(H):
        for w in range(W):
            if img[h][w] > s and f_img[h][w] > j:
                result[h][w] = 255
    print(result.dtype)
    return result


def vs(tag, re, dot_size = 0,
                                    inplace: bool = False) -> np.ndarray:
    """
    在uint8类型的NumPy数组（源图像）中寻找非零像素，在目标NumPy数组（目标图像）对应位置绘制红色点
    :param tag: 源图像的NumPy数组（uint8，灰度图shape=(H,W) / 彩色图shape=(H,W,3)），用于寻找非零像素
    :param re: 目标图像的NumPy数组（灰度图/彩色图均可），用于绘制红色点
    :param dot_size: 红色点的大小（默认2像素，半径）
    :param inplace: 是否在目标数组原地修改（默认False：复制后绘制，不影响原数组）
    :return: 绘制红色点后的目标图像NumPy数组
    """

    target_processed = re.copy() if not inplace else re
    if len(target_processed.shape) == 2:  # 灰度图 -> 彩色图（BGR格式）
        target_processed = cv2.cvtColor(target_processed, cv2.COLOR_GRAY2BGR)

    nonzero_mask = tag > 0

    # 获取非零像素的坐标（y: 行索引，x: 列索引）
    nonzero_y, nonzero_x = np.where(nonzero_mask)

    # 4. 在目标数组对应位置绘制红色点（OpenCV BGR格式：红色=(0,0,255)）
    red_color = (0, 0, 255)
    thickness = -1  # 填充式绘制（实心点）

    # 批量绘制（比循环更高效，尤其非零像素多时）
    for x, y in zip(nonzero_x, nonzero_y):
        cv2.circle(
            img=target_processed,
            center=(x, y),  # OpenCV绘图坐标：(列, 行) = (x, y)
            radius=dot_size,
            color=red_color,
            thickness=thickness
        )

    return target_processed


def peak_suppress_transform(img, g_min, g_max, inplace=False):
    """
    削峰变换（对应论文方法）:
        灰度 < g_min 的像素统一压到 g_min
        灰度 > g_max 的像素统一压到 g_max
        中间 [g_min, g_max] 不变
    用于让暗弱星点 + 背景噪声的双峰落在 [g_min, g_max] 的中间区域，然后在这个
    区间上做二维 Otsu 寻找最优阈值。:contentReference[oaicite:1]{index=1}

    参数
    ----
    img : np.ndarray, 灰度图，任意数值类型
    g_min : float/int, 削峰下界
    g_max : float/int, 削峰上界
    inplace : bool, True 则在原数组上改，False 则返回副本

    返回
    ----
    out : np.ndarray, 与 img 同 dtype
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img 必须是 numpy.ndarray")

    if g_min >= g_max:
        raise ValueError("g_min 必须小于 g_max")

    out = img if inplace else img.copy()

    # 为了避免整数溢出，用 float 计算再转回原类型
    out_f = out.astype(np.float32)

    out_f[out_f < g_min] = g_min
    out_f[out_f > g_max] = g_max

    return out_f.astype(img.dtype)

def rescale_gray(tag: np.ndarray, new_min: float, new_max: float, dtype=None):
    """
    将灰度值范围为 [0, 1] 的图像线性映射到 [new_min, new_max]

    Parameters
    ----------
    tag : np.ndarray
        输入灰度图，要求数值范围在 [0, 1]
    new_min : float
        目标最小灰度值
    new_max : float
        目标最大灰度值
    dtype : np.dtype or None
        输出数据类型，例如 np.uint8、np.float32
        若为 None，则保持浮点类型

    Returns
    -------
    np.ndarray
        映射后的灰度图
    """
    if new_max <= new_min:
        raise ValueError("new_max 必须大于 new_min")

    # 线性映射
    out = tag * (new_max - new_min) + new_min

    # 可选类型转换
    if dtype is not None:
        out = out.astype(dtype)

    return out

def mask_to_star_csv(mask, intensity, csv_path, min_area=3, connectivity=8):
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=connectivity)

    stars = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[label]
        pts = intensity[labels == label]
        flux = float(pts.sum())
        peak = float(pts.max()) if pts.size else 0.0

        stars.append({"id": len(stars)+1, "x": float(cx), "y": float(cy),
                      "area": int(area), "flux": flux, "peak": peak})

    df = pd.DataFrame(stars).sort_values("flux", ascending=False).reset_index(drop=True)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    return df

def save_centroids_to_csv(centroids):
    """
    保存灰度质心到CSV文件
    :param centroids: 质心数据，格式为 [(x, y), ...]
    :param save_path: 保存路径
    """
    save_path_npy = os.path.join(save_dir_centroids, "centroids.npy")
    save_path_csv = os.path.join(save_dir_centroids, "centroids.csv")

    # 确保文件夹存在，如果不存在则创建
    os.makedirs(save_dir_centroids, exist_ok=True)

    # 保存为npy文件（适用于快速加载）
    np.save(save_path_npy, centroids)
    # 保存到CSV文件
    df_centroids = pd.DataFrame(centroids, columns=['centroid_x', 'centroid_y'])
    df_centroids.to_csv(save_path_csv, index=False)

    print(f"质心数据已保存到：{save_path_npy} 和 {save_path_csv}")

def centroid_hash_u64(x_f4: np.float32, y_f4: np.float32) -> np.uint64:
    b = struct.pack("<ff", float(np.float32(x_f4)), float(np.float32(y_f4)))  # 固定 float32 序列化
    d = hashlib.blake2b(b, digest_size=8).digest()  # 8 bytes = 64-bit
    return np.frombuffer(d, dtype=np.uint64)[0]

def main_p1():
    # img = Image.open(image_path)
    # img = np.array(img)
    if FITS_PATH is not None:
        img, header, stats = load_fits_float32(FITS_PATH)

        print(stats)
        if stats["truncated_warning"]:
            print("注意：FITS 可能截断，建议重新拷贝/重新导出该文件。")

    raw_img = img.copy()

    print(f"图像形状: {img.shape}")
    print(f"像素统计: min={img.min():.1f}, max={img.max():.1f}, median={np.median(img):.1f}, std={np.std(img):.1f}")
    # img = np.array(img)

    print(f"图像尺寸: {img.shape}, 最小值={img.min():.1f}, 最大值={img.max():.1f}", {img.dtype})
    img_median = cv2.medianBlur(img, ksize=3)
    img_gaussian = cv2.GaussianBlur(img_median, ksize=(5, 5), sigmaX=1.0, sigmaY=1.0)
    img = img_gaussian.copy()
    print(f"滤波后，图像尺寸: {img.shape}, 最小值={img.min():.1f}, 最大值={img.max():.1f}", {img.dtype})
    np.save(os.path.join(save_dir, "img_gaussian.npy"), img_gaussian)
    save_path = os.path.join(save_dir, "img_gaussian.png")
    raw_img_save_path = os.path.join(save_dir, "raw_img.png")
    if FITS == False:
        Image.fromarray(img_gaussian).save(save_path)
    else:
        save_float_as_png_uint16(img_gaussian, save_path)
        save_float_as_png_uint16(raw_img, raw_img_save_path)
    # img = raw_img.copy()

    # cv2.imshow("PreImage", img)
    # cv2.imshow("3x3 Median Filter", img_median)
    # cv2.imshow("3x3 Gaussian Filter (sigma=1.0)", img_gaussian)

    # img = peak_suppress_transform(img, g_min=PEAK_TRANSFORM_MIN, g_max=PEAK_TRANSFORM_MAX).astype(np.float32)
    img_show = img.copy()
    I_D = img.copy()
    I_D_raw = img_gaussian.astype(np.float32)  # 滤波后但未削峰
    np.save(os.path.join(save_dir, "I_D_raw.npy"), I_D_raw)
    print(f"峰值变换后，图像尺寸: {img.shape}, 最小值={img.min():.1f}, 最大值={img.max():.1f}", {img.dtype})


    p_ij = compute_2d_hist(img)
    visualize_3d_hist(p_ij, save_path=save_dir)
    # 假设已经得到二维直方图 p_ij
    P = compute_prefix_sum(p_ij)

    s = S_INIT
    t = T_INIT
    L = p_ij.shape[0]  # 而不是 data_max+1

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
    mask_u8 = apply_2d_otsu_threshold(I_D, s_star, t_star)  # 用 I_D 更语义清晰

    # === 统一 mask 为 bool，并保存 ===

    mask_bool = (mask_u8 > 0)
    mask01 = mask_bool.astype(np.uint8)  # 0/1 uint8
    print(f"mask_u8: min={mask_u8.min()}, max={mask_u8.max()}, dtype={mask_u8.dtype}")
    print("mask pixels:", int(mask_bool.sum()))

    Image.fromarray(mask_u8).save(os.path.join(save_dir, "mask.png"))


    # 显示结果
    cv2.imshow("2D Otsu mask", mask_u8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grayscale_centroids = calculate_grayscale_centroid(mask_bool, raw_img)
    # for star in grayscale_centroids:
    #     print(f"Star {star['label']} - Centroid: ({star['centroid_x']}, {star['centroid_y']})")

    if FITS == False:
        # png图像计算精确率和召回率
        tag_img = Image.open(tag_img_path)
        tag_img_array = np.array(tag_img.convert('L'))
        tag_mask_bool = tag_img_array > 0
        geometric_centroids = calculate_geometric_centroid(tag_mask_bool)
        # 计算欧氏距离
        distances = match_centroids(geometric_centroids, grayscale_centroids)

        # 输出欧氏距离
        for dist in distances:
            if dist["distance"] is not None:
                print(f"Star {dist['label']} - Euclidean Distance: {dist['distance']:.2f}")
            else:
                print(f"Star {dist['label']} - Missing in grayscale centroids.")

        precision, recall = calculate_precision_recall(distances, geometric_centroids, grayscale_centroids)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
    # elif FITS == True:
    #     # 将已知星点作为标准，将计算精确率和召回率
    #     geometric_centroids = load_geometric_centroids_from_tag_csv(TAG_FITS_PATH)
    #     distances = match_centroids_fits(geometric_centroids, grayscale_centroids)

    #     for dist in distances:
    #         if dist["distance"] is not None:
    #             print(
    #                 f"Star {dist['label']} - Euclidean Distance: {dist['distance']:.2f} (matched_gray={dist.get('matched_gray_label')})")
    #         else:
    #             print(f"Star {dist['label']} - Missing / no close match in grayscale centroids.")

    #     precision, recall = calculate_precision_recall_fits(distances, geometric_centroids, grayscale_centroids)
    #     print(f"FITS_Precision: {precision:.2f}")
    #     print(f"FITS_Recall: {recall:.2f}")

    #     pd.DataFrame(distances).to_csv(os.path.join(save_dir, "centroid_match_fits.csv"), index=False)

    centroids = [(star['centroid_x'], star['centroid_y']) for star in grayscale_centroids]
    save_centroids_to_csv(centroids)

    os.makedirs(save_dir_centroids, exist_ok=True)

    # =========================
    # 1) 保存全量灰度质心（按 flux 已从大到小排好）
    # =========================
    allN = len(grayscale_centroids)
    all_hash = np.empty(
        (allN,),
        dtype=[
            ("hash", "u8"),
            ("x", "f4"),
            ("y", "f4"),
            ("flux", "f8"),
            ("area", "i4"),
        ]
    )

    all_x = np.array([star["centroid_x"] for star in grayscale_centroids], dtype=np.float32)
    all_y = np.array([star["centroid_y"] for star in grayscale_centroids], dtype=np.float32)
    all_flux = np.array([star["flux"] for star in grayscale_centroids], dtype=np.float64)
    all_area = np.array([star["area"] for star in grayscale_centroids], dtype=np.int32)

    all_hash["x"] = all_x
    all_hash["y"] = all_y
    all_hash["flux"] = all_flux
    all_hash["area"] = all_area
    all_hash["hash"] = np.array(
        [centroid_hash_u64(x, y) for x, y in zip(all_x, all_y)],
        dtype=np.uint64
    )

    np.savez_compressed(
        os.path.join(save_dir_centroids, "centroid_all.npz"),
        all_hash=all_hash
    )
    # 同时保存 centroid_all.csv
    csv_path_all = os.path.join(save_dir_centroids, "centroid_all.csv")
    with open(csv_path_all, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "hash", "x", "y", "flux", "area"])
        for i in range(allN):
            writer.writerow([
                i + 1,
                int(all_hash["hash"][i]),
                float(all_hash["x"][i]),
                float(all_hash["y"][i]),
                float(all_hash["flux"][i]),
                int(all_hash["area"][i]),
            ])

    # =========================
    # 2) 保存前 TOP_NUM 个（兼容你原来的流程）
    # =========================
    topN = min(TOP_NUM, allN)

    top_hash = np.empty(
        (topN,),
        dtype=[("hash", "u8"), ("x", "f4"), ("y", "f4")]
    )

    top_hash["x"] = all_x[:topN]
    top_hash["y"] = all_y[:topN]
    top_hash["hash"] = all_hash["hash"][:topN]

    np.savez_compressed(
        os.path.join(save_dir_centroids, "centroid_top.npz"),
        top_hash=top_hash
    )

    return grayscale_centroids

if __name__ == '__main__':
    centroids = main_p1()

