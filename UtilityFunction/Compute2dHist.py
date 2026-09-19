import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from getimages_cv import demo_on_fits
FITS_PATH = "20240306204703518_059051_01_L/20240306204801957_6002.fits"

def compute_2d_hist(image, L=None):
    """
    image: 输入 2D 数组，可为 float，但会转换成 int
    L: 灰度级数量，如果 None，则自动设为最大像素值+1
    """

    # --- 确保 image 可用于直方图索引 ---
    img_int = image.astype(np.int32)

    # 自动决定灰度级
    if L is None:
        L = img_int.max() + 1
    print("Using L =", L)

    # --- 邻域平均 ---
    kernel = np.ones((7, 7), np.float32) / 49
    mean_img = cv2.filter2D(image.astype(np.float32), -1, kernel)

    # 转成 int 索引
    mean_int = np.clip(mean_img, 0, L-1).astype(np.int32)

    # --- 构建直方图 ---
    H = np.zeros((L, L), dtype=np.float64)

    for i in range(img_int.shape[0]):
        for j in range(img_int.shape[1]):
            g = img_int[i, j]
            m = mean_int[i, j]
            H[g, m] += 1

    return H / H.sum()


def visualize_2d_hist(H):
    plt.figure(figsize=(8, 7))
    plt.imshow(np.log(H + 1e-12), cmap='hot', origin='lower')
    plt.title("2D Otsu Histogram (log scale)")
    plt.xlabel("Neighborhood mean (m)")
    plt.ylabel("Pixel gray (g)")
    plt.colorbar(label="log probability")
    plt.show()
def visualize_3d_hist(H, save_path=None):
    L = H.shape[0]

    g = np.arange(L)
    m = np.arange(L)
    G, M = np.meshgrid(g, m, indexing='ij')

    # 展平 (N, )
    X = G.flatten()
    Y = M.flatten()
    Z = H.flatten()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 柱子尺寸
    dx = dy = 1
    dz = Z

    # 颜色映射按高度来
    # colors = plt.cm.hot(Z / Z.max())


    # 颜色映射按高度来 + 低于0.05显示白色
    threshold = 0.05
    norm_val = Z.max() if Z.max() != 0 else 1  # 防止除0

    # 给每个柱子生成对应颜色，低于阈值=白色
    colors = []
    for z_val in Z:
        if z_val < threshold:
            colors.append([1, 1, 1, 1])  # 白色 RGBA
        else:
            # 高于阈值使用 hot 颜色映射
            colors.append(plt.cm.hot(z_val / norm_val))
    colors = np.array(colors)
    # ===================================================================

    ax.bar3d(X, Y, np.zeros_like(Z), dx, dy, dz, color=colors, shade=True)

    ax.set_xlabel("Gray level g")
    ax.set_ylabel("Neighborhood mean m")
    ax.set_zlabel("Frequency")

    plt.title("3D Histogram for 2D Otsu (g, m, freq)")
    if save_path != None:
        save_path = save_path + "\hist3d.png"
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] 3D histogram saved to: {save_path}")
    plt.show()

if __name__ == '__main__':
    beam_fwhm_pix = 3.1
    X_fwhm_pix = 10.0
    M_iter = 30

    img, data_max = demo_on_fits(
        fits_path=FITS_PATH,
        beam_fwhm_pix=beam_fwhm_pix,
        X_fwhm_pix=X_fwhm_pix,
        ext=0,
        M=M_iter,
        crop=(256, 256)
    )
    H = compute_2d_hist(img)
    visualize_3d_hist(H)