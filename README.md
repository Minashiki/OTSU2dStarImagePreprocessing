# OldStarImgPro

面向天基星图（FITS）的预处理与星点提取项目。项目的主流程使用固定分箱二维 Otsu 阈值分割，支持精确搜索和九宫格近似搜索，并可在批处理时识别、修复跨帧持续出现的孤立坏点。

除主流程外，`UtilityFunction/` 还提供质心计算、二维直方图分析，以及基于 Tycho 星表构建和处理三角形特征库的工具。

## 功能概览

- 读取二维 FITS 图像，并处理本设备的 `int16`/`uint16` 数据约定。
- 使用局部均值与固定分箱二维直方图进行二维 Otsu 阈值搜索。
- 提供 `exact`（精确）和 `nine-grid`（九宫格近似）两种搜索方法。
- 提供 `recall`、`balanced`、`purity` 三种阈值档位。
- 批处理多帧 FITS，估计持续性孤立坏点并导出掩膜、统计和星点质心。
- 生成掩膜、叠加图、二维/可选三维直方图及 CSV/NPY/NPZ 结果。

## 环境准备（Conda）

建议使用 Conda 隔离环境。以下命令在项目根目录执行：

```bash
conda create -n oldstarimgpro python=3.10 -y
conda activate oldstarimgpro
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

项目依赖中包含 OpenCV、Astropy 和 SciPy；如 Conda 环境已配置镜像或使用 `conda-forge`，也可按本机规范安装这些包。运行代码前请确认已激活 `oldstarimgpro` 环境。

## 快速开始

主入口为 `P1Preprocessing02.py`。仓库附带示例 FITS 数据，处理单个文件：

```bash
python P1Preprocessing02.py \
  --input "rst19/rst19/20260330163205413_9901.fits" \
  --output "rst19/outputs/demo_single" \
  --preset balanced \
  --search exact
```

处理目录内所有 FITS 文件：

```bash
python P1Preprocessing02.py \
  --batch \
  --input "rst19/rst19/*.fits" \
  --output "rst19/outputs/demo_batch" \
  --preset balanced \
  --search exact
```

Windows PowerShell 中可将上述命令写在一行，或将反斜杠续行符替换为反引号（`` ` ``）。路径包含空格时请保留双引号。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--input` | 单个 FITS 文件路径，或批处理使用的通配符。 |
| `--output` | 单帧输出目录，或批处理输出根目录。 |
| `--batch` | 强制按通配符批量处理；输入包含 `*`、`?` 或 `[` 时也会自动批处理。 |
| `--preset` | 阈值档位：`recall`、`balanced`（默认）或 `purity`。 |
| `--search` | 搜索方法：`exact`（默认）或 `nine-grid`。 |
| `--all-presets` | 额外导出全部三种档位的结果。 |
| `--signed` | 禁用相机 `int16` 到 `uint16` 的按位恢复；仅在输入确实是有符号数据时使用。 |
| `--interactive` | 单帧处理完成后显示所选掩膜。 |
| `--emit-3d` | 额外保存安全尺寸的三维直方图。 |

## 输出说明

单帧输出目录通常包含：

- `mask.png`：当前 `--preset` 与 `--search` 选择的二值星点掩膜。
- `overlay_*.png`：掩膜叠加在原图上的预览图。
- `stats.json`：输入、背景、阈值搜索和连通域统计。
- `Centroid_Top/`：星点质心结果（CSV、NPY、NPZ）。
- `hist2d_balanced.png` 等：二维直方图和各档位中间结果。

批处理还会在输出根目录生成：

- `bad_pixel_mask.npy` / `bad_pixel_mask.png`：自动识别的持续性坏点掩膜。
- `bad_pixel_candidates.csv`：候选坏点及出现次数。
- `threshold_compare.csv`：逐帧、逐档位的阈值与搜索结果汇总。

## 星表与特征库工具

`UtilityFunction/` 中的工具主要面向 SQLite 特征库。计算三角形特征主轴和投影值时，请显式传入数据库路径，避免使用脚本中的开发机默认绝对路径：

```bash
python UtilityFunction/ComputeStarLibraryAxis.py --db "D:/data/feature_triangle_database.db"
python UtilityFunction/ComputeStarLibraryP.py --db "D:/data/feature_triangle_database.db" --mode centered
```

上述脚本会在数据库中创建或更新表、列和索引；请先备份生产数据。

## 测试

在已激活的 Conda 环境中执行：

```bash
python -m pytest -q
```

## 项目结构

```text
P1Preprocessing02.py        主预处理命令行入口
SpaceStarOtsu.py            固定分箱二维 Otsu 核心算法
getimages_cv.py             FITS 读取和图像处理辅助函数
show_fits.py                FITS 可视化辅助函数
UtilityFunction/            质心、星表和特征库相关工具
rst19/                      示例 FITS 数据及处理结果
tests/                      自动化测试
```

## 注意事项

- FITS 文件可能较大，批处理前请确保有足够的磁盘空间保存掩膜、图像和数组结果。
- 默认的无符号恢复逻辑针对当前相机数据约定；不匹配的数据应使用 `--signed`，并先用少量样本确认灰度范围。
- `exact` 的结果更可复现，但速度通常低于 `nine-grid`；大批量数据可先比较两种方法的 `stats.json` 和汇总 CSV。
- 运行脚本时请从项目根目录启动，以确保 `UtilityFunction` 模块能够被正确导入。
