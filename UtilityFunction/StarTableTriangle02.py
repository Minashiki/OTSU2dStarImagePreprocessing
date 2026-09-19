import sqlite3
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

# ========= 1) 路径配置 =========
STAR_DB = "navigation_database.db"          # 你的星表db
TRI_DB  = "C:\\Users\Administrator\Desktop\北理工二月\北理工\PreProcess\\notes\StarTable\\feature_triangle_database.db"    # 新的特征三角形db

# 每颗主星查询多少个近邻（>=3；越大越稳，但越慢）
K_NEIGHBOR = 10

# 每颗主星用近邻里前两个来构一个三角形（当前你要求：最近两颗邻星）
# 如果你后续想“一主星+多组三角形”，这里可以扩展
USE_TWO_NEAREST = True

# ========= 2) 从星库读取数据 =========
conn = sqlite3.connect(STAR_DB)
cur = conn.cursor()
cur.execute("SELECT id, ra, dec FROM stars")
rows = cur.fetchall()
conn.close()

ids = np.array([r[0] for r in rows], dtype=object)
ra_deg  = np.array([r[1] for r in rows], dtype=np.float64)
dec_deg = np.array([r[2] for r in rows], dtype=np.float64)

# ========= 3) RA/Dec -> 单位球面向量 =========
ra  = np.deg2rad(ra_deg)
dec = np.deg2rad(dec_deg)
xyz = np.column_stack([
    np.cos(dec) * np.cos(ra),
    np.cos(dec) * np.sin(ra),
    np.sin(dec)
]).astype(np.float64)

# ========= 4) KDTree (一次性) =========
tree = cKDTree(xyz)

# ========= 5) 创建特征三角形库表（存排序后的三边） =========
conn_t = sqlite3.connect(TRI_DB)
cur_t = conn_t.cursor()

cur_t.execute("""
CREATE TABLE IF NOT EXISTS triangles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    center_id   TEXT,
    neighbor1_id TEXT,
    neighbor2_id TEXT,
    l1 REAL,   -- 排序后最短边
    l2 REAL,   -- 排序后中边
    l3 REAL    -- 排序后最长边
)
""")

# 建索引：后续查中心星、邻星更快；以后做匹配也方便
cur_t.execute("CREATE INDEX IF NOT EXISTS idx_tri_center ON triangles(center_id)")
cur_t.execute("CREATE INDEX IF NOT EXISTS idx_tri_n1     ON triangles(neighbor1_id)")
cur_t.execute("CREATE INDEX IF NOT EXISTS idx_tri_n2     ON triangles(neighbor2_id)")

conn_t.commit()

# ========= 6) 计算 + 三边排序 + 批量写入 =========
BATCH = 5000
buffer = []

for i in tqdm(range(len(xyz)), desc="Building triangles"):
    # query 返回：距离(欧氏) + 索引；球面上用向量点积算角距更合适
    dists, idxs = tree.query(xyz[i], k=K_NEIGHBOR)

    # idxs[0] 通常是自己
    # 取最近两颗邻星
    n1 = idxs[1]
    n2 = idxs[2]

    v0 = xyz[i]
    v1 = xyz[n1]
    v2 = xyz[n2]

    # 角距 = arccos(dot)
    a = np.arccos(np.clip(v0 @ v1, -1.0, 1.0))
    b = np.arccos(np.clip(v0 @ v2, -1.0, 1.0))
    c = np.arccos(np.clip(v1 @ v2, -1.0, 1.0))

    # 三边排序：l1<=l2<=l3
    l1, l2, l3 = np.sort([a, b, c])

    buffer.append((ids[i], ids[n1], ids[n2], float(l1), float(l2), float(l3)))

    if len(buffer) >= BATCH:
        cur_t.executemany(
            "INSERT INTO triangles(center_id, neighbor1_id, neighbor2_id, l1, l2, l3) VALUES (?, ?, ?, ?, ?, ?)",
            buffer
        )
        conn_t.commit()
        buffer.clear()

# flush
if buffer:
    cur_t.executemany(
        "INSERT INTO triangles(center_id, neighbor1_id, neighbor2_id, l1, l2, l3) VALUES (?, ?, ?, ?, ?, ?)",
        buffer
    )
    conn_t.commit()
    buffer.clear()

# ========= 7) 简单验证 =========
cur_t.execute("SELECT center_id, neighbor1_id, neighbor2_id, l1, l2, l3 FROM triangles LIMIT 5")
print("Sample triangles:")
for r in cur_t.fetchall():
    print(r)

conn_t.close()
