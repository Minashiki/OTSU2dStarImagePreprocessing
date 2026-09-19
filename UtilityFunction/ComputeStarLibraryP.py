"""
ComputeStarLibraryP.py

在 feature_triangle_database.db 中：
1) 从 p_axis(id=1) 读取 μ 与 Ω
2) 给 triangles 表增加 p 列（如果不存在）
3) 分批读取 triangles 的 (id,l1,l2,l3)，计算
       P = Ω^T ( [l1,l2,l3] - μ )
4) 分批回写到 triangles.p

运行：
    python ComputeStarLibraryP.py --db feature_triangle_database.db

可选：
    --table triangles
    --batch 200000
    --mode centered    # centered: Ω^T (X-μ)；raw: Ω^T X

注意：
- 这个脚本会对 100 万行做 UPDATE，建议放在 SSD 上运行。
"""

import os
import argparse
import sqlite3
import numpy as np
from tqdm import tqdm


def column_exists(cur, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]  # (cid, name, type, notnull, dflt_value, pk)
    return col in cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",
                    default=r"C:\Users\Administrator\Desktop\北理工二月\北理工\PreProcess\notes\StarTable\feature_triangle_database.db",
                    help="特征三角形数据库路径")
    ap.add_argument("--table", default="triangles", help="三角形表名（默认 triangles）")
    ap.add_argument("--batch", type=int, default=200000, help="每批处理行数（默认 200000）")
    ap.add_argument("--mode", choices=["centered", "raw"], default="centered",
                    help="P 的定义：centered=Ω^T(X-μ)（推荐），raw=Ω^T X（书面公式）")
    ap.add_argument("--commit_every", type=int, default=1,
                    help="每处理多少个 batch 提交一次（默认 1；更大可能更快但崩溃风险更大）")
    args = ap.parse_args()

    db_path = args.db
    table = args.table
    batch = args.batch
    mode = args.mode
    commit_every = max(1, args.commit_every)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"找不到数据库文件：{db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1) 读取 μ 与 Ω
    cur.execute("SELECT mu1, mu2, mu3, w1, w2, w3 FROM p_axis WHERE id=1")
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("找不到 p_axis(id=1)。请先运行 ComputeStarLibraryAxis.py 生成 μ 和 Ω。")

    mu = np.array(row[0:3], dtype=np.float64)
    omega = np.array(row[3:6], dtype=np.float64)

    # 归一化保险
    omega_norm = np.linalg.norm(omega)
    if omega_norm == 0:
        raise RuntimeError("p_axis 中的 Ω 向量范数为 0，数据异常。")
    omega = omega / omega_norm

    print("[INFO] Using mu =", mu)
    print("[INFO] Using omega (normalized) =", omega)
    print("[INFO] P mode =", mode)

    # 2) 增加 p 列（如不存在）
    if not column_exists(cur, table, "p"):
        print("[INFO] Column 'p' not found. Adding it ...")
        cur.execute(f"ALTER TABLE {table} ADD COLUMN p REAL")
        conn.commit()
        print("[INFO] Column 'p' added.")
    else:
        print("[INFO] Column 'p' already exists. Will overwrite p values.")

    # 可选：给 p 建索引（后续用 P 做检索会很重要）
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_p ON {table}(p)")
    conn.commit()

    # 3) 总行数
    cur.execute(f"SELECT COUNT(1) FROM {table}")
    total_n = int(cur.fetchone()[0])
    print(f"[INFO] triangles total rows = {total_n}")

    # 4) 分批读取 + 回写
    offset = 0
    batch_i = 0

    pbar = tqdm(total=total_n, desc="Computing & updating P")

    while True:
        cur.execute(
            f"SELECT id, l1, l2, l3 FROM {table} LIMIT ? OFFSET ?",
            (batch, offset)
        )
        rows = cur.fetchall()
        if not rows:
            break

        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        X = np.array([(r[1], r[2], r[3]) for r in rows], dtype=np.float64)  # (m,3)

        if mode == "centered":
            Xc = X - mu
            p = Xc @ omega  # (m,)
        else:
            p = X @ omega

        # executemany: UPDATE table SET p=? WHERE id=?
        params = list(zip(p.astype(float).tolist(), ids.astype(int).tolist()))
        cur.executemany(f"UPDATE {table} SET p=? WHERE id=?", params)

        offset += len(rows)
        batch_i += 1
        pbar.update(len(rows))

        if batch_i % commit_every == 0:
            conn.commit()

    conn.commit()
    pbar.close()

    # 5) 快速抽样验证
    cur.execute(f"SELECT id, l1, l2, l3, p FROM {table} ORDER BY id LIMIT 5")
    print("\n[INFO] Sample rows with p:")
    for r in cur.fetchall():
        print(r)

    conn.close()
    print("\n[DONE] 已计算并写入 triangles.p。后续可用 p 做索引检索。")


if __name__ == "__main__":
    main()
