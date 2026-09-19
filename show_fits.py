import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize

def show_fits_zscale(path, figsize=(8, 8), crop=None, cmap='gray'):
    """
    用 ZScale + Asinh 拉伸显示 FITS 图像（类似 DS9 默认显示效果）

    参数:
    ----------
    path : str
        FITS 文件路径
    figsize : tuple, optional
        图像显示窗口大小 (宽, 高)，默认 (8, 8)
    crop : tuple or None, optional
        如果希望只显示中心区域，可传 (h_crop, w_crop)，例如 (512, 512)
        表示裁剪中心 512×512 区域
    cmap : str, optional
        颜色映射，默认 'gray'
    """
    # 1️⃣ 读取 FITS 数据
    hdul = fits.open(path)
    data = hdul[0].data.astype(np.float32)
    hdul.close()

    # 2️⃣ 如果需要裁剪中心区域
    if crop is not None:
        h, w = data.shape
        ch, cw = crop
        data = data[h//2 - ch//2 : h//2 + ch//2,
                    w//2 - cw//2 : w//2 + cw//2]

    # 3️⃣ 使用 ZScaleInterval + AsinhStretch 计算显示范围
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())

    # 4️⃣ 显示图像
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap=cmap, origin='lower', norm=norm)
    plt.title("ZScale + Asinh 显示 (仿 DS9)", fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04, label='Pixel Value')
    plt.tight_layout()
    plt.show()



# def show_images_grid(images, titles=None, ncols=4, cmap='gray', norm=None, figsize=(12, 8)):
#     """
#     通用图像展示函数 —— 根据输入的图像数量自动生成子图网格。
#
#     Parameters
#     ----------
#     images : list[np.ndarray]
#         图像列表，例如 [img1, img2, img3, ...]
#     titles : list[str], optional
#         对应的标题名称，与 images 一一对应。
#     ncols : int, default=4
#         每行显示的列数。
#     cmap : str, default='gray'
#         图像颜色映射表。
#     norm : ImageNormalize or None
#         可选的 astropy 可视化归一化（如 zscale_asinh(img)）。
#     figsize : tuple
#         整个图的大小。
#     """
#     n = len(images)
#     if titles is None:
#         titles = [f"Image {i+1}" for i in range(n)]
#
#     nrows = int(np.ceil(n / ncols))
#
#     plt.figure(figsize=figsize)
#     for i, (img, title) in enumerate(zip(images, titles)):
#         plt.subplot(nrows, ncols, i + 1)
#         plt.imshow(img, cmap=cmap, origin='lower', norm=norm)
#         plt.title(title)
#         plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np

def zscale_asinh(img):
    return ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())

def show_images_grid(
    images,
    titles=None,
    ncols=4,
    cmap='gray',
    norm=None,
    figsize=None,
    crop=None
):
    """
    通用图像展示函数 —— 自动根据图像数量调整画布大小，并支持中心裁剪显示。

    Parameters
    ----------
    images : list[np.ndarray]
        图像列表，例如 [img1, img2, img3, ...]
    titles : list[str], optional
        每张图的标题。
    ncols : int, default=4
        每行显示的列数。
    cmap : str, default='gray'
        图像颜色映射表。
    norm : ImageNormalize or None
        可选的 astropy 归一化显示（如 zscale_asinh(img)）。
    figsize : tuple or None
        若为 None，则根据行列自动计算自适应大小。
    crop : tuple(int, int) or None
        若指定 (ch, cw)，则裁剪每张图中心区域大小 (ch × cw) 进行显示。
    """
    n = len(images)
    if n == 0:
        print("⚠️ 没有图像输入！")
        return

    if titles is None:
        titles = [f"Image {i+1}" for i in range(n)]

    # 计算行列
    nrows = int(np.ceil(n / ncols))

    # === 自适应画布大小 ===
    if figsize is None:
        fig_w = min(4 * ncols, 18)
        fig_h = min(4 * nrows, 14)
        figsize = (fig_w, fig_h)

    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        # --- 若指定 crop，则裁剪中心区域 ---
        if crop is not None:
            h, w = img.shape[:2]
            ch, cw = crop
            y1 = max(h // 2 - ch // 2, 0)
            y2 = min(h // 2 + ch // 2, h)
            x1 = max(w // 2 - cw // 2, 0)
            x2 = min(w // 2 + cw // 2, w)
            img_display = img[y1:y2, x1:x2]
        else:
            img_display = img

        ax = plt.subplot(nrows, ncols, i + 1)
        # ax.imshow(img_display, cmap=cmap, origin='lower', norm=norm)
        ax.imshow(img_display, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()




# 调用示例
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from show_fits import show_images_grid, zscale_asinh
# if __name__ == '__main__':
    # # 示例 1：默认显示完整图
    # show_images_grid(
    #     [img, gauss_noisy, mean_gauss, gauss_gauss],
    #     ["raw", "noisy", "mean", "gaussian"],
    #     ncols=2,
    #     norm=zscale_asinh(img)
    # )
#
# # 示例 2：仅显示中心 256×256 区域
# show_images_grid(
#     [img, gauss_noisy, mean_gauss, gauss_gauss],
#     ["raw", "noisy", "mean", "gaussian"],
#     ncols=2,
#     norm=zscale_asinh(img),
#     crop=(256, 256)
# )