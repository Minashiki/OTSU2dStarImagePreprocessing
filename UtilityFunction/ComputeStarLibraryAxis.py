"""
ComputeStarLibraryAxis.py

从 feature_triangle_database.db 的 triangles 表读取 (l1,l2,l3)，
在线计算均值 μ 和协方差矩阵 Z（中心化二阶矩），求最大特征值对应的特征向量 Ω，
并写入 p_axis 表（仅一行，id=1）。

运行：
    python ComputeStarLibraryAxis.py --db feature_triangle_database.db

可选参数：
    --table triangles   (默认 triangles)
    --batch 200000      (默认 200000)
"""

import os
import argparse
import sqlite3
import numpy as np
from tqdm import tqdm


def online_mean_cov_3d(rows_iter, total_n: int):
    """
    Welford 在线算法计算 3D 均值与协方差累计矩阵 M2 (3x3)。
    返回：
        mu: (3,)
        cov: (3,3)  # 使用总体协方差：M2 / N
        n:  使用样本数
    """
    mu = np.zeros(3, dtype=np.float64)
    M2 = np.zeros((3, 3), dtype=np.float64)
    n = 0

    for x in tqdm(rows_iter, total=total_n, desc="Scanning triangles (online mean/cov)"):
        x = np.array(x, dtype=np.float64)  # (l1,l2,l3)
        n += 1
        delta = x - mu
        mu += delta / n
        delta2 = x - mu
        M2 += np.outer(delta, delta2)

    if n == 0:
        raise RuntimeError("triangles 表为空：没有任何 (l1,l2,l3) 数据可用于计算 μ, Ω")

    cov = M2 / n  # 总体协方差
    return mu, cov, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=r"C:\Users\Administrator\Desktop\北理工二月\北理工\PreProcess\notes\StarTable\feature_triangle_database.db", help="特征三角形数据库路径")
    ap.add_argument("--table", default="triangles", help="三角形表名（默认 triangles）")
    ap.add_argument("--batch", type=int, default=200000, help="每批读取行数（默认 200000）")
    args = ap.parse_args()

    db_path = args.db
    table = args.table
    batch = args.batch

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"找不到数据库文件：{db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1) 总行数（进度条用）
    cur.execute(f"SELECT COUNT(1) FROM {table}")
    total_n = int(cur.fetchone()[0])
    print(f"[INFO] triangles total rows = {total_n}")

    # 2) 分批读取生成器（避免一次性占满内存）
    def rows_generator():
        offset = 0
        while True:
            cur.execute(
                f"SELECT l1, l2, l3 FROM {table} LIMIT ? OFFSET ?",
                (batch, offset)
            )
            rows = cur.fetchall()
            if not rows:
                break
            for r in rows:
                yield r
            offset += len(rows)

    mu, cov, n_used = online_mean_cov_3d(rows_generator(), total_n=total_n)

    # 3) 最大特征值对应特征向量 = 最优投影主轴 Ω
    eigvals, eigvecs = np.linalg.eigh(cov)  # cov 对称，eigh 更稳
    axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float64)

    # 统一方向（可选）：让最大绝对值分量为正，保证复现一致
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis

    # 归一化（保险）
    axis = axis / np.linalg.norm(axis)

    print("\n[RESULT] mu (mean of [l1,l2,l3]) =")
    print(mu)
    print("[RESULT] omega (principal axis Ω) =")
    print(axis)
    print("[RESULT] max eigenvalue =", float(np.max(eigvals)))

    # 4) 写入 p_axis 表（只存一行，id=1）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS p_axis (
        id INTEGER PRIMARY KEY,
        mu1 REAL, mu2 REAL, mu3 REAL,
        w1 REAL, w2 REAL, w3 REAL
    )
    """)

    cur.execute(
        "INSERT OR REPLACE INTO p_axis (id, mu1, mu2, mu3, w1, w2, w3) VALUES (1, ?, ?, ?, ?, ?, ?)",
        (float(mu[0]), float(mu[1]), float(mu[2]), float(axis[0]), float(axis[1]), float(axis[2]))
    )
    conn.commit()

    # 5) 读回校验
    cur.execute("SELECT id, mu1, mu2, mu3, w1, w2, w3 FROM p_axis WHERE id=1")
    row = cur.fetchone()
    print("\n[INFO] p_axis saved (id=1):")
    print(row)

    conn.close()
    print("\n[DONE] 已将 μ 和 Ω 写入 p_axis 表。后续计算 P 时请固定使用该表中的参数。")


if __name__ == "__main__":
    main()
