import numpy as np
import math
from UtilityFunction.CameraParams import camera_params, CAM


def select_neighbors(main_star, centroids, radius_range=(1.0, 7000.0)):
    neighbors = []
    for i, (x, y) in enumerate(centroids):
        if i != main_star:
            distance = np.linalg.norm([x - centroids[main_star][0], y - centroids[main_star][1]])  # 计算欧氏距离
            if radius_range[0] <= distance <= radius_range[1]:
                neighbors.append((x, y))  # 只保留坐标部分
    neighbors.sort(key=lambda x: np.linalg.norm([x[0] - centroids[main_star][0], x[1] - centroids[main_star][1]]))  # 按距离升序排序
    return neighbors[:2]  # 返回前两个邻星

def pixel_to_unit_ray(x, y, cam=CAM):
    """
    像素坐标 (x,y) -> 相机坐标系单位视线向量 u
    """
    X = (x - cam["cx"]) * cam["dx"]
    Y = (y - cam["cy"]) * cam["dy"]
    Z = cam["f"]
    v = np.array([X, Y, Z], dtype=np.float64)
    return v / np.linalg.norm(v)

def angular_distance_pixels(p1, p2, cam=CAM):
    """
    两个像素点之间的严格角距（弧度）
    p1,p2: (x,y)
    """
    u1 = pixel_to_unit_ray(p1[0], p1[1], cam)
    u2 = pixel_to_unit_ray(p2[0], p2[1], cam)
    return float(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))

def triangle_edges_sorted(pA, pB, pC, cam=CAM):
    """
    三角形三边角距（弧度）+ 排序：l1<=l2<=l3
    """
    a = angular_distance_pixels(pA, pB, cam)
    b = angular_distance_pixels(pB, pC, cam)
    c = angular_distance_pixels(pC, pA, cam)
    l1, l2, l3 = np.sort([a, b, c])
    return float(l1), float(l2), float(l3)


def construct_all_feature_triangles(centroids):
    all_feature_triangles = []
    feature_vectors = []
    # 遍历每颗主星
    for main_star_index in range(len(centroids)):
        main_star = centroids[main_star_index]

        # 选择主星对应的两个邻星
        neighbors = select_neighbors(main_star_index, centroids)
        neighbor1 = neighbors[0]
        neighbor2 = neighbors[1]

        # 计算角距
        l1, l2, l3 = triangle_edges_sorted(main_star, neighbor1, neighbor2)
        character_vector = (l1, l2, l3)

        feature_vectors.append(character_vector)
        # 存储特征三角形信息
        all_feature_triangles.append({
            'main_star': main_star,
            'neighbor1': neighbor1,
            'neighbor2': neighbor2,
            'character_vector': character_vector
        })

    return all_feature_triangles, feature_vectors




def best_projection_axis_pca(V):
    """
    V: (N,3) 每行 [l1,l2,l3]
    返回: mu(3,), axis(3,)
    """
    V = np.asarray(V, dtype=np.float64)
    mu = V.mean(axis=0)
    X = V - mu
    C = (X.T @ X) / max(len(V), 1)

    eigvals, eigvecs = np.linalg.eigh(C)
    axis = eigvecs[:, np.argmax(eigvals)]  # 第一主轴

    # 统一方向（可选）
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    np.savez(
        "notes/p_axis.npz",
        mu=mu,
        axis=axis
    )

    return mu, axis

def project_p(v, mu, axis):
    """
    v: (3,) -> P值（标量）
    """
    v = np.asarray(v, dtype=np.float64)
    return float(axis @ (v - mu))


if __name__ == '__main__':
    from CameraParams import camera_params, CAM
    # 示例星点数据
    centroids = [(1, 2), (4, 6), (3, 1), (7, 8), (6, 3)]  # 这里是(x, y)格式的星点坐标

    # 构建特征三角形
    all_feature_triangles, feature_vectors = construct_all_feature_triangles(centroids)
    print(feature_vectors)

    # 输出结果
    # for i, feature_triangle in enumerate(all_feature_triangles):
        # print(f"特征三角形 {i + 1}:")
        # print(f"主星坐标: {feature_triangle['main_star']}")
        # print(f"邻星坐标1: {feature_triangle['neighbor1']}")
        # print(f"邻星坐标2: {feature_triangle['neighbor2']}")
        # print(f"主星和邻星之间的角度: {feature_triangle['angle']:.2f}°")
        # print(f"特征向量: {feature_triangle['character_vector']}")
    # print(feature_vectors)
    # feature_vectors: list of (l1,l2,l3)
    V = np.array(feature_vectors, dtype=np.float64)

    mu, axis = best_projection_axis_pca(V)

    p_values = [project_p(v, mu, axis) for v in V]
    print("check l1<=l2<=l3:", np.all(V[:, 0] <= V[:, 1] + 1e-12) and np.all(V[:, 1] <= V[:, 2] + 1e-12))
    print("l1 min/max:", V[:, 0].min(), V[:, 0].max())
    print("l3 min/max:", V[:, 2].min(), V[:, 2].max())
