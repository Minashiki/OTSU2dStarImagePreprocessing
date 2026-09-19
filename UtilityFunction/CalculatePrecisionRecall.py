def calculate_precision_recall(distances, geometric_centroids, grayscale_centroids):
    # True Positives (TP): 欧氏距离在阈值范围内的匹配
    TP = sum(1 for dist in distances if dist["distance"] is not None)

    # False Positives (FP): 模拟图像中存在，但没有找到对应几何质心的星点
    FP = 0
    for gray_star in grayscale_centroids:
        # 对于每颗模拟图像中的星点，检查它是否在标签图像中有对应的几何质心
        if not any(geo_star['label'] == gray_star['label'] and dist['distance'] is not None
                   for dist, geo_star in zip(distances, geometric_centroids) if dist['label'] == gray_star['label']):
            FP += 1

    # False Negatives (FN): 标签图像中存在的星点，在模拟图像中没有找到匹配
    FN = 0
    for geo_star in geometric_centroids:
        # 对于每颗标签图像中的星点，检查它是否在模拟图像中找到了匹配的灰度质心
        if not any(dist['label'] == geo_star['label'] and dist['distance'] is not None
                   for dist in distances):
            FN += 1

    # 计算精确率和召回率
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall

def calculate_precision_recall02(distances, geometric_centroids, grayscale_centroids):
    tp = sum(1 for d in distances if d.get("distance") is not None)
    fp = max(0, len(grayscale_centroids) - tp)
    fn = max(0, len(geometric_centroids) - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def calculate_precision_recall_fits(distances, geometric_centroids, grayscale_centroids):
    TP = sum(1 for d in distances if d.get("distance", None) is not None)

    # FP: 灰度质心里没被任何 geo 匹配到的
    FP = 0
    for gray_star in grayscale_centroids:
        if not any(
            (d.get("matched_gray_label", None) == gray_star.get("label", None)) and (d.get("distance", None) is not None)
            for d in distances
        ):
            FP += 1

    # FN: geo 里没匹配到灰度质心的
    FN = 0
    for geo_star in geometric_centroids:
        if not any(
            (d.get("label", None) == geo_star.get("label", None)) and (d.get("distance", None) is not None)
            for d in distances
        ):
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall


# 示例用法
# if __name__ == '__main__':
#     # 计算欧氏距离
#     distances = match_centroids(geometric_centroids, grayscale_centroids)
#
#     # 输出欧氏距离
#     for dist in distances:
#         if dist["distance"] is not None:
#             print(f"Star {dist['label']} - Euclidean Distance: {dist['distance']:.2f}")
#         else:
#             print(f"Star {dist['label']} - Missing in grayscale centroids.")
#
#     # 计算精确率和召回率
#     precision, recall = calculate_precision_recall(distances)
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
