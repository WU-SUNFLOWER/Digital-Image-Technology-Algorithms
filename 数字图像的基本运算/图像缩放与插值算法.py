import cv2
import numpy as np

# 最近邻插值
def nearest_neighbor_resize(image, x, y):
    height, width, channels = image.shape

    x = min(round(x), width - 1)
    y = min(round(y), height - 1)

    return image[y, x]

# 双线性插值
def bilinear_interpolate(image, x, y):
    x1, y1 = int(x), int(y)
    height, width, channels = image.shape
    
    # 越界检查
    if x1 >= width - 1 and y1 < height - 1:
        return image[y1, -1]
    elif x1 < width - 1 and y1 >= height - 1:
        return image[-1, x1]
    elif x1 >= width - 1 and y1 >= height - 1:
        return image[-1, -1]
        
    """
    (x1, y1)-------------(x2, y1)
       |                    |
       |      (x, y)        |
       |                    |
    (x1, y2)-------------(x2, y2)
    """
    
    x2, y2 = x1 + 1, y1 + 1
    Q11 = image[y1, x1]
    Q21 = image[y1, x2]
    Q12 = image[y2, x1]
    Q22 = image[y2, x2]
    
    # 水平方向进行双线性插值
    R1 = (x2 - x) / (x2 - x1) * Q11 + (x - x1) / (x2 - x1) * Q21
    R2 = (x2 - x) / (x2 - x1) * Q12 + (x - x1) / (x2 - x1) * Q22
    
    # 垂直方向进行单次线性插值
    P = (y2 - y) / (y2 - y1) * R1 + (y - y1) / (y2 - y1) * R2
    
    return P

def my_resize(image, scale_factor, interpolator):
    # 获取原图像的尺寸和通道数
    height, width, channels = image.shape
    
    # 创建新图像的numpy数组
    new_width = round(width * scale_factor)
    new_height = round(height * scale_factor)
    resized_image = np.zeros((new_height, new_width, channels), np.uint8)
    
    # 对每个新图像的像素位置应用插值算法
    for i in range(new_height):
        for j in range(new_width):
            y = i / scale_factor
            x = j / scale_factor
            resized_image[i, j] = interpolator(image, x, y)
                
    return resized_image

def main():
    image_path = 'test.jpg'  # 更换为你的图片路径
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 缩放图像
    resized_image1 = my_resize(image, 2, nearest_neighbor_resize)
    resized_image2 = my_resize(image, 2, bilinear_interpolate)

    # 保存结果
    result = np.hstack([resized_image1, resized_image2, resized_image3])
    cv2.imwrite("result.jpg", result)

if __name__ == '__main__':
    main()