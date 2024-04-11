import cv2
import numpy as np

def convolve(image, kernel, target_row, target_col, delta_rows, delta_cols):
    total_rows, total_cols = image.shape
    top = max(0, target_row - delta_rows)
    bottom = min(total_rows - 1, target_row + delta_rows)
    left = max(0, target_col - delta_cols)
    right = min(total_cols - 1, target_col + delta_cols)
    
    result = 0
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
          result += image[row, col] * kernel[row - top, col - left]
    
    return result

def create_gaussian_kernel(delta_rows, delta_cols, sigma = 5):
    size_width = 2 * delta_cols + 1
    size_height = 2 * delta_rows + 1
    kernel = np.zeros((size_width, size_height), dtype=np.float32)
    
    first_term = 1 / np.sqrt(2 * np.pi) / sigma
    second_term = lambda x, y: np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    
    for row in range(0, size_height):
        for col in range(0, size_width):
            kernel[row, col] = \
                first_term * second_term(row - delta_rows, col - delta_cols)
            
    # 归一化
    kernel /= np.sum(kernel)
    return kernel

def create_neighborhood_kernel(delta_rows, delta_cols):
    size_width = 2 * delta_cols + 1
    size_height = 2 * delta_rows + 1
    kernel = np.full((size_width, size_height), 1.0, dtype=np.float32)
    kernel /= np.sum(kernel)
    return kernel

def myblur(image, ksize, kernel_creator):
    # 获取图像的大小
    total_rows, total_cols = image.shape

    # 创建一个和输入图像一样大小的数组，用于存储结果
    result = np.zeros((total_rows, total_cols), dtype=np.float32)

    # 计算卷积核
    delta_cols, delta_rows = ksize[0] // 2, ksize[1] // 2
    kernel = kernel_creator(delta_rows, delta_cols)

    # 对每个像素进行卷积运算
    for row in range(total_rows):
        for col in range(total_cols):
            result[row, col] = \
                convolve(image, kernel, 
                            row, col, delta_rows, 
                            delta_cols)

    # 将结果转换为合适的数据类型
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

# 读取图像
image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

# 测试邻域平均法
blurred_image_neighborhood = \
    myblur(image, (3, 3), create_neighborhood_kernel)
# 测试高斯模糊
blurred_image_gaussian = \
    myblur(image, (3, 3), create_gaussian_kernel)

# 输出结果
images = np.hstack([image, blurred_image_neighborhood, blurred_image_gaussian])
cv2.imwrite("output.png", images)