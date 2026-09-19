import sqlite3
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# 读取星表
conn = sqlite3.connect("navigation_database.db")
cur = conn.cursor()
cur.execute("SELECT id, ra, dec FROM stars")
stars = cur.fetchall()
conn.close()

ids = np.array([s[0] for s in stars])
ra = np.deg2rad(np.array([s[1] for s in stars]))
dec = np.deg2rad(np.array([s[2] for s in stars]))

# 球面单位向量
xyz = np.column_stack([
    np.cos(dec) * np.cos(ra),
    np.cos(dec) * np.sin(ra),
    np.sin(dec)
])

# 建 KDTree（一次）
tree = cKDTree(xyz)
conn_tri = sqlite3.connect("feature_triangle_database.db")
cur_tri = conn_tri.cursor()

cur_tri.execute("""
CREATE TABLE IF NOT EXISTS triangles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    center_id TEXT,
    neighbor1_id TEXT,
    neighbor2_id TEXT,
    edge1 REAL,
    edge2 REAL,
    edge3 REAL
)
""")

K = 10  # 查找最近 10 颗邻星

for i in tqdm(range(len(xyz)), desc="Building triangles"):
    dists, idxs = tree.query(xyz[i], k=K)

    # 跳过自身 idx=0
    n1, n2 = idxs[1], idxs[2]

    v0 = xyz[i]
    v1 = xyz[n1]
    v2 = xyz[n2]

    # 三条角距（弧度）
    a = np.arccos(np.clip(v0 @ v1, -1, 1))
    b = np.arccos(np.clip(v0 @ v2, -1, 1))
    c = np.arccos(np.clip(v1 @ v2, -1, 1))

    cur_tri.execute(
        "INSERT INTO triangles (center_id, neighbor1_id, neighbor2_id, edge1, edge2, edge3) VALUES (?, ?, ?, ?, ?, ?)",
        (ids[i], ids[n1], ids[n2], a, b, c)
    )

conn_tri.commit()
conn_tri.close()
