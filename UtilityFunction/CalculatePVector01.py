import numpy as np
import math
from UtilityFunction.CameraParams import camera_params


def select_neighbors(main_star, centroids, radius_range=(1.0, 7000.0)):
    neighbors = []
    for i, (x, y) in enumerate(centroids):
        if i != main_star:
            distance = np.linalg.norm([x - centroids[main_star][0], y - centroids[main_star][1]])  # 计算欧氏距离
            if radius_range[0] <= distance <= radius_range[1]:
                neighbors.append((x, y))  # 只保留坐标部分
    neighbors.sort(key=lambda x: np.linalg.norm([x[0] - centroids[main_star][0], x[1] - centroids[main_star][1]]))  # 按距离升序排序
    return neighbors[:2]  # 返回前两个邻星


# 计算像素坐标到角度的转换函数
def pixel_to_angle(x, y, focal, cen_x, cen_y, dx, dy):
    # 计算归一化坐标
    x_normalized = (x - cen_x) * dx
    y_normalized = (y - cen_y) * dy

    # 计算角度
    angle_x = math.atan(x_normalized / focal)
    angle_y = math.atan(y_normalized / focal)

    return angle_x, angle_y


# 计算两个点之间的角距
def angle_distance(p1, p2):
    # p1 和 p2 是图像中的两个点 (x1, y1), (x2, y2)
    angle_x1, angle_y1 = pixel_to_angle(p1[0], p1[1], camera_params['Focal'], camera_params['CenX'],
                                        camera_params['CenY'], camera_params['DX'], camera_params['DY'])
    angle_x2, angle_y2 = pixel_to_angle(p2[0], p2[1], camera_params['Focal'], camera_params['CenX'],
                                        camera_params['CenY'], camera_params['DX'], camera_params['DY'])

    # 计算两个点的角距 (通过球面余弦法则)
    delta_x = angle_x2 - angle_x1
    delta_y = angle_y2 - angle_y1
    angle_dist = math.sqrt(delta_x ** 2 + delta_y ** 2)

    return angle_dist


# 计算三角形匹配的角距
def match_triangle_angles(points):
    # points 是一个包含3个点的列表 [(x1, y1), (x2, y2), (x3, y3)]
    angle12 = angle_distance(points[0], points[1])
    angle23 = angle_distance(points[1], points[2])
    angle31 = angle_distance(points[2], points[0])

    # 返回三角形的三个角距
    return angle12, angle23, angle31

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
        angle1, angle2, angle_between_neighbors = match_triangle_angles([main_star, neighbor1, neighbor2])

        # 特征向量
        character_vector = (angle1, angle2, angle_between_neighbors)
        feature_vectors.append(character_vector)
        # 存储特征三角形信息
        all_feature_triangles.append({
            'main_star': main_star,
            'neighbor1': neighbor1,
            'neighbor2': neighbor2,
            'character_vector': character_vector
        })

    return all_feature_triangles, feature_vectors


# 计算所有三维向量的均值（投影点的均值）
def calculate_projection_mean(vectors):
    vectors_array = np.array(vectors)
    mean_vector = np.mean(vectors_array, axis=0)
    return mean_vector


# 计算投影点的标准偏差
def calculate_projection_std(vectors, mean_vector):
    vectors_array = np.array(vectors)
    deviations = np.linalg.norm(vectors_array - mean_vector, axis=1)
    return np.std(deviations)


# 计算投影的误差（基于公式3-7）
def calculate_projection_error(vectors, mean_vector):
    vectors_array = np.array(vectors)
    error = np.mean(np.linalg.norm(vectors_array - mean_vector, axis=1) ** 2)
    return error


# 优化投影轴的函数（基于公式3-9）
def optimize_projection_axis(vectors):
    # 计算投影点的均值
    mean_vector = calculate_projection_mean(vectors)

    # 计算投影点的标准偏差
    std_dev = calculate_projection_std(vectors, mean_vector)

    # 计算投影误差
    error = calculate_projection_error(vectors, mean_vector)

    # 这里可以通过优化方法进一步优化投影轴
    # 在这个简单示例中我们假设投影轴就是平均向量的方向
    projection_axis = mean_vector / np.linalg.norm(mean_vector)

    return projection_axis, mean_vector, error, std_dev


if __name__ == '__main__':
    from CameraParams import camera_params
    # 示例星点数据
    centroids = [(1, 2), (4, 6), (3, 1), (7, 8), (6, 3)]  # 这里是(x, y)格式的星点坐标

    # 构建特征三角形
    all_feature_triangles, feature_vectors = construct_all_feature_triangles(centroids)

    # 输出结果
    # for i, feature_triangle in enumerate(all_feature_triangles):
        # print(f"特征三角形 {i + 1}:")
        # print(f"主星坐标: {feature_triangle['main_star']}")
        # print(f"邻星坐标1: {feature_triangle['neighbor1']}")
        # print(f"邻星坐标2: {feature_triangle['neighbor2']}")
        # print(f"主星和邻星之间的角度: {feature_triangle['angle']:.2f}°")
        # print(f"特征向量: {feature_triangle['character_vector']}")
    # print(feature_vectors)
    projection_axis, mean_vector, error, std_dev = optimize_projection_axis(feature_vectors)
    print(f"最佳投影轴方向：{projection_axis}")
    print(f"投影点的均值：{mean_vector}")
    print(f"投影点的标准偏差：{std_dev}")
    print(f"投影误差：{error}")