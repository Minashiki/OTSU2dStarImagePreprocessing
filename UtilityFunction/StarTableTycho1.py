import sqlite3
import scipy.io

# 加载Tycho1数据
mat_file_path = 'C:\\Users\Administrator\Desktop\北理工1月\北理工\PreProcess\Tycho1\Tycho1.mat'  # 修改为你的文件路径
mat_data = scipy.io.loadmat(mat_file_path)

# 获取Tycho1星表中的数据
tycho_data = mat_data['Tyc']

# 连接数据库
conn = sqlite3.connect('navigation_database.db')
cursor = conn.cursor()

# 确保表格已经创建
cursor.execute('''CREATE TABLE IF NOT EXISTS stars (
                    id TEXT PRIMARY KEY,    -- 星体编号（使用TEXT类型以便存储较长的编号）
                    magnitude REAL,            -- 视星等（亮度）
                    ra REAL,                   -- 赤经（RA）
                    dec REAL                   -- 赤纬（Dec）
                    );''')

# 从Tycho1数据导入星表数据
for star in tycho_data:
    # 获取编号（假设编号由3部分组成，拼接成一个唯一标识符）
    id = f"{int(star[0]):05d}{int(star[1]):05d}{int(star[2]):05d}"  # 将编号拼接成一个唯一ID
    magnitude = star[3]         # 视星等
    ra = star[4]                # 赤经（RA）
    dec = star[5]               # 赤纬（Dec）

    try:
        # 插入数据到星表
        cursor.execute("INSERT INTO stars (id, magnitude, ra, dec) VALUES (?, ?, ?, ?)",
                       (id, magnitude, ra, dec))
        # print("star: ", id)
    except sqlite3.IntegrityError:
        # 如果遇到重复的ID，则跳过该行数据
        print(f"Skipping duplicate star ID: {id}")

# 提交更改
conn.commit()

# 查询并显示星表数据
cursor.execute("SELECT * FROM stars LIMIT 10")  # 只查看前10行
stars = cursor.fetchall()
print("Stars in Database:")
for star in stars:
    print(star)

# 关闭连接
conn.close()
