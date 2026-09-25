# OTSU2d

面向天基星图（FITS）的预处理与星点提取项目。项目的主流程使用固定分箱二维 Otsu 阈值分割，支持精确搜索和九宫格近似搜索，并可在批处理时识别、修复跨帧持续出现的孤立坏点。

除主流程外，`UtilityFunction/` 还提供质心计算、二维直方图分析，以及基于 Tycho 星表构建和处理三角形特征库的工具。

## 功能概览

- 读取二维 FITS 图像，并处理本设备的 `int16`/`uint16` 数据约定。
- 使用局部均值与固定分箱二维直方图进行二维 Otsu 阈值搜索。
- 可选的分块模式：滤波后将图像切分为任意大小的矩形块，每块独立求解二维 Otsu 阈值对并分割前景/背景，边界不足一块时保持原样、不补全。
- 提供 `exact`（精确）和 `nine-grid`（九宫格近似）两种搜索方法。
- 提供 `recall`、`balanced`、`purity` 三种阈值档位。
- 批处理多帧 FITS，估计持续性孤立坏点并导出掩膜、统计和星点质心。
- 生成掩膜、叠加图、二维/可选三维直方图及 CSV/NPY/NPZ 结果。
- 可选地把管线中间数据原样导出为 FITS（`--save-fits`），不做 PNG 的分位数裁剪与拉伸，便于与原生 FITS 图像直接对比审查。

## 环境准备（Conda）

建议使用 Conda 隔离环境。以下命令在项目根目录执行：

```bash
conda create -n otsu2d python=3.10 -y
conda activate otsu2d
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

项目依赖中包含 OpenCV、Astropy 和 SciPy；如 Conda 环境已配置镜像或使用 `conda-forge`，也可按本机规范安装这些包。运行代码前请确认已激活 `otsu2d` 环境。

## 快速开始

主入口为 `P1Preprocessing02.py`。仓库附带示例 FITS 数据，处理单个文件：

```bash
python P1Preprocessing02.py \
  --input "ImageData/1.fits" \
  --output "output/demo_single" \
  --preset balanced \
  --search exact
```

处理目录内所有 FITS 文件：

```bash
python P1Preprocessing02.py \
  --batch \
  --input "ImageData/*.fits" \
  --output "output/demo_batch" \
  --preset balanced \
  --search exact
```

启用分块模式（每块 50×50，块间完全独立）：

```bash
python P1Preprocessing02.py \
  --input "ImageData/1.fits" \
  --output "output/demo_block50" \
  --preset balanced \
  --search exact \
  --block-size 50
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
| `--block-size` | 可选分块大小，如 `50`（50×50）或 `50x80`。设置后滤波图像被切分为矩形块，每块独立估计背景、求解二维 Otsu 阈值对并分割前景/背景；边界不足一块时保持原尺寸、不补全。未指定时保持整幅处理。 |
| `--signed` | 禁用相机 `int16` 到 `uint16` 的按位恢复；仅在输入确实是有符号数据时使用。 |
| `--interactive` | 单帧处理完成后显示所选掩膜。 |
| `--emit-3d` | 额外保存安全尺寸的三维直方图。 |
| `--save-fits` | 额外把各 PNG 对应的管线原始数组保存为同名 FITS 文件（如 `img_gaussian.fits`、`mask.fits`）。FITS 保留 float32/uint8 原值，不做 PNG 的分位数裁剪与拉伸，因此 `img_gaussian.fits` 与建立二维直方图的数据完全一致。 |

## 输出说明

单帧输出目录通常包含：

- `mask.png`：当前 `--preset` 与 `--search` 选择的二值星点掩膜。
- `overlay_*.png`：掩膜叠加在原图上的预览图。
- `stats.json`：输入、背景、阈值搜索和连通域统计。分块模式下阈值相关字段为各块的均值，并带有 `block_count`、`grid_shape`、`nine_grid_unstable_blocks` 等汇总信息。
- `Centroid_Top/`：星点质心结果（CSV、NPY、NPZ）。
- `hist2d_balanced.png` 等：二维直方图和各档位中间结果（仅整幅模式；分块模式下各块直方图上界不同，不生成该图）。

分块模式（`--block-size`）还会额外生成：

- `blocks_<preset>.csv`：逐块的尺寸、位置、直方图上界、两种搜索的阈值对与得分。
- `block_threshold_s_<preset>_<search>.npy/.png`、`block_threshold_t_<preset>_<search>.npy/.png`：按块网格排列的阈值分布图（NPY 数组及可视化 PNG）。

指定 `--save-fits` 时，上述 PNG 图像（`img_gaussian`、`raw_img`、`mask*`、`overlay_*` 及分块阈值分布图）会各多一份同名 `.fits` 文件：灰度图为 float32 原值、掩膜为 uint8 原值、叠加图为通道在前的三通道 uint8（NAXIS3=3），均未经过 PNG 的压缩与像素值拉伸，可直接与原生 FITS 图像对比审查。

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
- 分块模式逐块独立求解，4096×4096 图像配 50×50 块会产生 82×82=6724 次搜索，耗时明显高于整幅模式；批处理汇总 CSV 中分块档位的阈值字段为各块均值。
- 运行脚本时请从项目根目录启动，以确保 `UtilityFunction` 模块能够被正确导入。
