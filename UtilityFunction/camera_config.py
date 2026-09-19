import yaml
import math
import numpy as np

class CameraConfig:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # self.ra_deg = cfg["attitude"]["ra_deg"]
        # self.dec_deg = cfg["attitude"]["dec_deg"]
        # self.roll_deg = cfg["attitude"]["roll_deg"]

        self.width = cfg["sensor"]["resolution"]["width"]
        self.height = cfg["sensor"]["resolution"]["height"]
        self.pixel_size_mm = cfg["sensor"]["pixel_size_mm"]
        self.focal_length_mm = cfg["sensor"]["focal_length_mm"]

        # 主点（默认图像中心）
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0

    # ---------- 派生量 ----------
    @property
    def fov_rad(self):
        return 2.0 * math.atan(
            (self.width * self.pixel_size_mm / 2.0) / self.focal_length_mm
        )

    @property
    def fov_deg(self):
        return math.degrees(self.fov_rad)

    # ---------- 像素 -> 单位向量 ----------
    def pixel_to_unit(self, x, y):
        X = (x - self.cx) * self.pixel_size_mm
        Y = (y - self.cy) * self.pixel_size_mm
        Z = self.focal_length_mm
        v = np.array([X, Y, Z], dtype=np.float64)
        n = np.linalg.norm(v)
        if n == 0:
            return None
        return v / n
