import numpy as np
import cv2
from scipy.spatial.distance import euclidean
import pandas as pd

# def calculate_grayscale_centroid(mask_bool, image):
#     """
#     计算每颗星的灰度质心
#     :param mask_bool: 布尔型的星点掩膜图像，True 表示星点区域
#     :param image: 原始图像，要求为灰度图（numpy array）
#     :return: 每颗星的灰度质心坐标列表
#     """
#     # 计算连通组件，提取每颗星的区域
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8))
#
#     # 存储每颗星的灰度质心
#     grayscale_centroids = []
#
#     for label in range(1, num_labels):  # 从1开始，跳过背景
#         # 提取每颗星对应的区域
#         star_mask = (labels == label)
#
#         # 获取该星区域的像素坐标
#         star_coords = np.argwhere(star_mask)  # (y, x) 格式的坐标列表
#
#         # 提取该星的灰度值
#         star_intensity = image[star_mask]
#         print("label", label)
#         print("star_intensity", star_intensity)
#
#         # 计算灰度质心
#         weighted_x = np.sum(star_coords[:, 1] * star_intensity)
#         weighted_y = np.sum(star_coords[:, 0] * star_intensity)
#         total_intensity = np.sum(star_intensity)
#
#         # 灰度质心
#         if total_intensity > 0:
#             centroid_x = weighted_x / total_intensity
#             centroid_y = weighted_y / total_intensity
#             grayscale_centroids.append({"label": label, "centroid_x": centroid_x, "centroid_y": centroid_y})
#
#     return grayscale_centroids

def calculate_grayscale_centroid(mask_bool, image):
    """
    计算每颗星的灰度质心，并按亮度(总灰度和 flux)从亮到暗排序
    :param mask_bool: bool 掩膜，True 为星点区域
    :param image: 灰度图 numpy array
    :return: grayscale_centroids: list[dict]，按亮到暗排序
             dict: {"label": int, "centroid_x": float, "centroid_y": float, "flux": float}
             如果你不想要 flux 字段，可以在返回前删掉（见下方注释）
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8))

    items = []
    for label in range(1, num_labels):  # 跳过背景 0
        star_mask = (labels == label)

        # 该星点区域的像素坐标 (y, x)
        ys, xs = np.where(star_mask)
        if xs.size == 0:
            continue

        # 强度（保证与坐标严格对齐）
        star_intensity = image[ys, xs].astype(np.float64)

        flux = float(star_intensity.sum())  # 亮度指标：总灰度和
        if flux <= 0:
            continue

        # 灰度加权质心
        centroid_x = float((xs * star_intensity).sum() / flux)
        centroid_y = float((ys * star_intensity).sum() / flux)
        area = stats[label, cv2.CC_STAT_AREA]

        items.append({
            "label": label,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "flux": flux,
            "area": area
        })

    # 按亮度从亮到暗排序
    items.sort(key=lambda d: d["flux"], reverse=True)

    return items


# def calculate_geometric_centroid(mask_bool):
#     """
#     计算标签图像中每颗星的几何质心
#     :param mask_bool: 标签图像的布尔掩膜，True 表示星点区域
#     :return: 每颗星的几何质心坐标列表
#     """
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8))
#     geometric_centroids = []
#     for label in range(1, num_labels):
#         # 获取几何质心坐标
#         cx, cy = centroids[label]
#         geometric_centroids.append({"label": label, "centroid_x": cx, "centroid_y": cy})
#
#     return geometric_centroids

def calculate_geometric_centroid(mask_bool):
    """
    计算标签图像中每颗星的几何质心，并计算每颗星的面积
    :param mask_bool: 标签图像的布尔掩膜，True 表示星点区域
    :return: 每颗星的几何质心坐标和面积
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8))
    geometric_centroids = []

    for label in range(1, num_labels):  # 跳过背景
        # 获取几何质心坐标和面积
        cx, cy = centroids[label]
        area = stats[label, cv2.CC_STAT_AREA]  # 获取星点的面积
        geometric_centroids.append({"label": label, "centroid_x": cx, "centroid_y": cy, "area": area})

    return geometric_centroids


# def match_centroids(geometric_centroids, grayscale_centroids, distance_threshold=10):
#     """
#     根据欧氏距离匹配几何质心和灰度质心，确保只有距离较近的质心被匹配
#     :param geometric_centroids: 标签图像中的几何质心列表
#     :param grayscale_centroids: 模拟图像中的灰度质心列表
#     :param distance_threshold: 欧氏距离阈值，只有当距离小于此阈值时，认为两个质心对应同一颗星
#     :return: 匹配后的欧氏距离列表
#     """
#     distances = []
#     used_gray_labels = set()  # 用于记录已匹配的灰度质心标签
#
#     for geo_star in geometric_centroids:
#         # 尝试在灰度质心中找到最接近的星点
#         closest_star = None
#         min_distance = float('inf')
#
#         for gray_star in grayscale_centroids:
#             if gray_star['label'] not in used_gray_labels:  # 避免重复匹配
#                 dist = euclidean((geo_star['centroid_x'], geo_star['centroid_y']),
#                                  (gray_star['centroid_x'], gray_star['centroid_y']))
#                 if dist < min_distance and dist < distance_threshold:  # 只匹配较近的星点
#                     min_distance = dist
#                     closest_star = gray_star
#
#         if closest_star:
#             distances.append({"label": geo_star['label'], "distance": min_distance})
#             used_gray_labels.add(closest_star['label'])
#         else:
#             distances.append({"label": geo_star['label'], "distance": None})  # 如果找不到匹配的，标记为缺失
#
#     return distances

def match_centroids(geometric_centroids, grayscale_centroids):
    """
    根据欧氏距离匹配几何质心和灰度质心，确保只有距离较近的质心被匹配
    匹配条件是：欧氏距离小于标签图像中这颗星的三分之二的面积大小
    :param geometric_centroids: 标签图像中的几何质心列表
    :param grayscale_centroids: 模拟图像中的灰度质心列表
    :return: 匹配后的欧氏距离列表
    """
    distances = []
    used_gray_labels = set()  # 用于记录已匹配的灰度质心标签

    for geo_star in geometric_centroids:
        # 获取该星的面积
        area = geo_star['area']
        # 计算最大匹配距离，假设三分之二的面积对应的最大距离
        max_distance = np.sqrt((2 / 3) * area)

        # 尝试在灰度质心中找到最接近的星点
        closest_star = None
        min_distance = float('inf')

        for gray_star in grayscale_centroids:
            if gray_star['label'] not in used_gray_labels:  # 避免重复匹配
                dist = euclidean((geo_star['centroid_x'], geo_star['centroid_y']),
                                 (gray_star['centroid_x'], gray_star['centroid_y']))
                if dist < min_distance and dist < max_distance:  # 只匹配较近的星点
                    min_distance = dist
                    closest_star = gray_star

        if closest_star:
            distances.append({"label": geo_star['label'], "distance": min_distance})
            used_gray_labels.add(closest_star['label'])
        else:
            distances.append({"label": geo_star['label'], "distance": None})  # 如果找不到匹配的，标记为缺失

    return distances

def _as_centroid_dict_list(centroids):
    out = []
    for i, c in enumerate(centroids):
        if isinstance(c, dict):
            d = dict(c)
            if "centroid_x" not in d and "x" in d:
                d["centroid_x"] = d["x"]
            if "centroid_y" not in d and "y" in d:
                d["centroid_y"] = d["y"]
            if "label" not in d:
                d["label"] = i + 1
            out.append(d)
        else:
            x, y = c
            out.append({"label": i + 1, "centroid_x": float(x), "centroid_y": float(y), "area": None})
    return out


def match_centroids_fits(geometric_centroids, grayscale_centroids):
    """
    根据欧氏距离匹配几何质心和灰度质心，1-1 匹配。
    匹配条件：距离 < (2/3) * area（area 取标签图像的那颗星；若没有 area 则用灰度质心的 area；还没有就用 5px 兜底）
    返回：list[dict]，每个元素至少包含 {"label":..., "distance":...}
    """
    # 兼容 geometric/grayscale 里可能是 tuple(x,y) 或 dict
    def _as_list(centroids):
        out = []
        for i, c in enumerate(centroids):
            if isinstance(c, dict):
                d = dict(c)
                if "centroid_x" not in d and "x" in d: d["centroid_x"] = d["x"]
                if "centroid_y" not in d and "y" in d: d["centroid_y"] = d["y"]
                if "label" not in d: d["label"] = i + 1
                out.append(d)
            else:
                x, y = c
                out.append({"label": i + 1, "centroid_x": float(x), "centroid_y": float(y), "area": None})
        return out

    geo_list = _as_list(geometric_centroids)
    gray_list = _as_list(grayscale_centroids)

    used_gray = set()
    distances = []

    for geo in geo_list:
        gx, gy = float(geo["centroid_x"]), float(geo["centroid_y"])
        geo_area = geo.get("area", None)

        best_j = None
        best_dist = None

        for j, gray in enumerate(gray_list):
            if j in used_gray:
                continue

            dx = gx - float(gray["centroid_x"])
            dy = gy - float(gray["centroid_y"])
            dist = float((dx*dx + dy*dy) ** 0.5)

            area = geo_area if geo_area is not None else gray.get("area", None)
            thr = (2.0/3.0) * float(area) if (area is not None and float(area) > 0) else 5.0

            if dist <= thr and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_j = j

        if best_j is None:
            distances.append({"label": int(geo.get("label", 0)), "distance": None})
        else:
            used_gray.add(best_j)
            distances.append({
                "label": int(geo.get("label", 0)),
                "distance": float(best_dist),
                "matched_gray_label": int(gray_list[best_j].get("label", 0))
            })

    return distances

def load_geometric_centroids_from_tag_csv(csv_path):
    df = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(csv_path)

    cols = list(df.columns)

    if "H" in cols and "I" in cols:
        xcol, ycol = "H", "I"
    elif "x" in cols and "y" in cols:
        xcol, ycol = "x", "y"
    elif "X" in cols and "Y" in cols:
        xcol, ycol = "X", "Y"
    else:
        if df.shape[1] < 9:
            raise ValueError(f"{csv_path} 列数不足 9，无法按 H/I 推断坐标列。实际列：{df.shape[1]}")
        xcol, ycol = df.columns[7], df.columns[8]  # H/I 对应第 8/9 列

    label_col = next((c for c in ("A", "id", "ID") if c in cols), None)
    area_col = next((c for c in ("J", "area", "Area", "AREA") if c in cols), None)

    xs = pd.to_numeric(df[xcol], errors="coerce")
    ys = pd.to_numeric(df[ycol], errors="coerce")
    labels = pd.to_numeric(df[label_col], errors="coerce") if label_col else pd.Series(np.arange(1, len(df) + 1), index=df.index)
    areas = pd.to_numeric(df[area_col], errors="coerce") if area_col else None

    geometric_centroids = []
    for idx in df.index:
        x = xs.loc[idx]
        y = ys.loc[idx]
        if np.isnan(x) or np.isnan(y):
            continue
        lab = labels.loc[idx]
        lab = int(lab) if not np.isnan(lab) else (len(geometric_centroids) + 1)
        area = None
        if areas is not None:
            a = areas.loc[idx]
            if not np.isnan(a):
                area = float(a)

        geometric_centroids.append({"label": lab, "centroid_x": float(x), "centroid_y": float(y), "area": area})

    return geometric_centroids

# # 示例用法
# if __name__ == '__main__':
#     # 加载标签图像（无噪声的正确星图）和模拟图像
#     tag_img = Image.open('path_to_tag_image.png')
#     tag_img_array = np.array(tag_img.convert('L'))  # 灰度化标签图像
#
#     # 计算标签图像的几何质心
#     tag_mask_bool = tag_img_array > 0  # 假设标签图像的非零部分表示星点
#     geometric_centroids = calculate_geometric_centroid(tag_mask_bool)
#
#     # 计算模拟图像的灰度质心
#     grayscale_centroids = calculate_grayscale_centroid(mask_bool, raw_img)
#
#     # 计算欧氏距离
#     distances = match_centroids(geometric_centroids, grayscale_centroids, distance_threshold=10)
#
#     # 输出欧氏距离
#     for dist in distances:
#         if dist["distance"] is not None:
#             print(f"Star {dist['label']} - Euclidean Distance: {dist['distance']:.2f}")
#         else:
#             print(f"Star {dist['label']} - Missing in grayscale centroids.")
