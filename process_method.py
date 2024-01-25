import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import convolve




def subtract_plane(Z):
    def fit_plane(x, y, z):
        """拟合平面并返回拟合参数"""
        A = np.c_[x, y, np.ones(x.shape)]
        C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)  # 求解最小二乘问题
        return C  # 返回平面参数
    """从每一行减去拟合的平面"""
    n, m = Z.shape
    for i in range(n):
        y = np.full(m, i)
        x = np.arange(m)
        C = fit_plane(x, y, Z[i, :])
        # 计算平面并从行中减去
        Z[i, :] -= (C[0] * x + C[1] * y + C[2])
    return Z



#################################################################################
def subtract_curve(z_clean):
    def fit_and_subtract(row):
        def quadratic_fit(x, a, b, c):
            """二次曲线函数"""
            return a * x ** 2 + b * x + c

        """对一行数据进行二次曲线拟合，并从原始数据中减去拟合曲线"""
        x = np.arange(len(row))
        params, _ = curve_fit(quadratic_fit, x, row)
        fitted_curve = quadratic_fit(x, *params)
        return row - fitted_curve
    z_processed=np.array([fit_and_subtract(row) for row in z_clean])
    return z_processed

#################################################################################
def invert(arr):
    """将数组中的最大值变为最小值，最小值变为最大值，以此类推"""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    inverted_arr = arr_max - (arr - arr_min)
    return inverted_arr



#################################################################################
def sharpen(image, alpha=1.0):
    """
    对二维数据（图像）进行锐化处理。

    :param image: 二维numpy数组，表示图像。
    :param alpha: 锐化强度，值越大锐化效果越强。
    :return: 锐化后的图像。
    """
    # 定义一个锐化核
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]]) * alpha + np.array([[0, 0, 0],
                                                                   [0, 1, 0],
                                                                   [0, 0, 0]])
    # 应用卷积进行锐化
    sharpened_image = convolve(image, sharpening_kernel, mode='nearest')
    return sharpened_image

#################################################################################
# def histogram_equalization(image):
#     # 由于直方图均衡化通常应用于像素值（这些值通常是整数），
#     # 因此需要先将浮点数数组转换为整数数组。此外，还需要确保转换后的值位于有效的像素值范围内
#     # （例如，对于8位图像，范围通常是0到255）
#     image = image.astype('uint8')
#     """
#     对二维数据（灰度图像）进行直方图均衡化。
#     :param image: 二维numpy数组，表示灰度图像。
#     :return: 直方图均衡化后的图像。
#     """
#     # 计算图像的直方图
#     hist, bins = np.histogram(image.flatten(), 256, [0, 256])
#
#     # 计算累积分布函数（CDF）
#     cdf = hist.cumsum()
#     cdf_normalized = cdf * hist.max() / cdf.max()
#
#     # 将CDF用于直方图均衡化
#     cdf_m = np.ma.masked_equal(cdf, 0)
#     cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
#     cdf = np.ma.filled(cdf_m, 0).astype('uint8')
#
#     # 将直方图均衡化映射应用于原始图像
#     equalized_image = cdf[image]
#
#     return equalized_image


def histogram_equalization(image):
    """
    对二维浮点数数组进行归一化和直方图均衡化。
    :param image: 二维浮点数numpy数组。
    :return: 直方图均衡化后的图像。
    """
    # 将图像归一化到0-1范围
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # 将归一化的图像转换为0-255的整数
    image_255 = (normalized_image * 255).astype('uint8')

    # 应用直方图均衡化
    hist, bins = np.histogram(image_255.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = 255 * cdf / cdf.max()
    equalized_image = cdf_normalized[image_255]

    # （可选）将均衡化后的图像转换回原始浮点数的范围
    equalized_image_float = (equalized_image / 255) * (np.max(image) - np.min(image)) + np.min(image)

    return equalized_image_float

