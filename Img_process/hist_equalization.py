import numpy as np
import matplotlib.pyplot as plt
# Set the font family to SimHei (黑体)
plt.rcParams['font.family'] = 'SimHei'

def numpy_hist_equalization(image_arr, bit_depth=8):
    """
    NumPy直方图均衡化 (支持任意位深)
    
    参数:
        image_arr: 输入图像numpy数组 (2D或3D)
        bit_depth: 图像位深度 (8/12/14/16)
    
    返回:
        equalized: 直方图均衡化后的图像数组
        mapping: 灰度映射函数 (用于分析)
    """
    # 输入验证
    assert isinstance(image_arr, np.ndarray), "输入必须为NumPy数组"
    ndim = image_arr.ndim
    assert ndim in [2, 3], "仅支持灰度图(2D)或RGB(3D)"
    
    # 计算最大灰度值
    L = 2**bit_depth - 1
    dtype = image_arr.dtype
    
    # RGB图像处理 (通道分离)
    if ndim == 3:
        channels = []
        mapping = []
        for c in range(image_arr.shape[2]):
            eq_channel, map_func = _equalize_channel(image_arr[:, :, c], L, dtype)
            channels.append(eq_channel)
            mapping.append(map_func)
        return np.stack(channels, axis=2), mapping
    
    # 灰度图像处理
    return _equalize_channel(image_arr, L, dtype)

def _equalize_channel(channel, L, dtype):
    """单通道均衡化核心函数"""
    # 计算原始直方图
    hist, bins = np.histogram(channel.flatten(), bins=L, range=(0, L))
    
    # 计算累积分布函数(CDF)
    cdf = hist.cumsum()
    
    # 避免除0错误和极小值问题
    cdf_min = cdf[np.nonzero(cdf)[0][0]] if np.any(cdf > 0) else 0
    n_pixels = channel.size
    
    # 计算映射函数
    mapping = np.zeros(L, dtype=np.float32)
    valid_idx = hist > 0
    mapping[valid_idx] = (cdf[valid_idx] - cdf_min) / (n_pixels - cdf_min + 1e-6) * L
    
    # 应用映射函数
    stretched = np.interp(channel, np.arange(L), mapping)
    
    # 保持原数据类型
    return stretched.astype(dtype), mapping

# ==================== 高级增强版 ====================
def adaptive_hist_equalization(image_arr, grid_size=8, clip_limit=0.03, bit_depth=8):
    """
    自适应直方图均衡化 (CLAHE变种)
    
    参数:
        grid_size: 局部窗口大小 (建议4-16)
        clip_limit: 直方图裁剪阈值 (0.01-0.05)
        bit_depth: 图像位深度
    
    返回:
        enhanced: 自适应均衡化结果
    """
    h, w = image_arr.shape[:2]
    L = 2**bit_depth - 1
    
    # 计算分块参数
    x_step = max(1, w // grid_size)
    y_step = max(1, h // grid_size)
    
    # 创建输出数组
    result = np.zeros_like(image_arr)
    
    # 处理每个分块
    for y in range(0, h, y_step):
        for x in range(0, w, x_step):
            # 提取分块 (处理边界)
            y1, y2 = y, min(y + y_step, h)
            x1, x2 = x, min(x + x_step, w)
            tile = image_arr[y1:y2, x1:x2]
            
            # 分块直方图均衡化
            tile_eq, _ = numpy_hist_equalization(tile, bit_depth)
            
            # 应用裁剪限制
            if clip_limit > 0:
                hist = np.histogram(tile.flatten(), bins=L, range=(0, L))[0]
                max_val = clip_limit * tile.size
                excess = np.sum(np.maximum(0, hist - max_val))
                if excess > 0:
                    # 直方图裁剪重分配
                    hist_clip = np.clip(hist, None, max_val)
                    redist = excess // L
                    hist_eq = hist_clip + redist
                    mapping = np.cumsum(hist_eq) / hist_eq.sum() * L
                    tile_eq = np.interp(tile, np.arange(L), mapping)
            
            # 写入结果
            result[y1:y2, x1:x2] = tile_eq
    
    # 双线性插值平滑分块边界
    if grid_size > 1:
        from scipy.ndimage import map_coordinates
        grid_points = np.mgrid[0:h, 0:w].reshape(2, -1).T
        map_coords = np.stack([grid_points[:, 0], grid_points[:, 1]], axis=1).T
        smoothed = map_coordinates(result, map_coords, order=1).reshape(h, w)
        return smoothed
    
    return result

# ==================== 物理优化功能 ====================
def detector_correction(image_arr, dark_current, flat_field, bit_depth=16):
    """
    探测器响应校正 (物理必做前置处理)
    公式: I_corr = (I_raw - dark_current) / flat_field
    """
    image_corr = np.maximum(0, image_arr - dark_current)
    image_corr = image_corr / (flat_field + 1e-9)
    # 动态范围压缩 (保持原始位深)
    return (L * (image_corr / np.max(image_corr))).astype(image_arr.dtype)

def photon_transfer_curve(image_arrs, bit_depth=16):
    """
    生成光子转移曲线 (评估探测效率)
    输入: 多个同场景不同曝光图像
    返回: 散粒噪声系数, 系统增益
    """
    means = np.mean([img.mean() for img in image_arrs])
    vars = np.var([img.flatten() for img in image_arrs], axis=1)
    fit = np.polyfit(means, vars, 1)
    gain = fit[0]  # 系统增益
    read_noise = np.mean([np.var(img) for img in image_arrs])  # 读出噪声
    return gain, read_noise

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建示例低对比度图像 (1024x1024, 12-bit)
    h, w = 1024, 1024
    bit_depth = 12
    L = 2**bit_depth - 1  # 4095
    
    # 生成带梯度背景的物理图像
    gradient = np.linspace(0, 0.7*L, w)[np.newaxis, :] * np.ones((h, 1))
    particles = np.random.rand(h, w) * 0.3*L
    sample = 1500 * np.exp(-(np.square(np.arange(h)[:,None]-h/2) + 
                           np.square(np.arange(w)-w/2))/(2*(w/8)**2))
    
    orig_img = np.clip(gradient + particles + sample, 0, L).astype(np.uint16)
    
    # 基础均衡化
    eq_img, mapping = numpy_hist_equalization(orig_img, bit_depth)
    
    # 自适应均衡化 (增强弱对比区域)
    adap_eq = adaptive_hist_equalization(orig_img, grid_size=8, clip_limit=0.02, bit_depth=bit_depth)
    
    # 结果分析
    def calc_contrast(img):
        """计算对比度指数 (物理有效性验证)"""
        roi1 = img[h//4:h//2, w//4:w//2].mean()  # 样品区
        roi2 = img[h//8*7:, w//8*7:].mean()      # 背景区
        return (roi1 - roi2) / (roi1 + roi2 + 1e-6)
    
    print(f"原始对比度: {calc_contrast(orig_img):.4f}")
    print(f"均衡对比度: {calc_contrast(eq_img):.4f}")
    print(f"自适应对比度: {calc_contrast(adap_eq):.4f}")
    
    # 可视化 (仅在小图像时启用)
    if h * w <= 10000000:  # 限制百万像素内
        plt.figure(figsize=(12, 8))
        plt.subplot(221), plt.imshow(orig_img, cmap='gray'), plt.title('原始图像')
        plt.subplot(222), plt.imshow(eq_img, cmap='gray'), plt.title('基础均衡化')
        plt.subplot(223), plt.imshow(adap_eq, cmap='gray'), plt.title('自适应均衡化')
        plt.subplot(224), plt.plot(np.arange(L)+1, mapping, 'r'), 
        plt.title('灰度映射函数'), plt.grid()
        plt.tight_layout(), plt.show()
