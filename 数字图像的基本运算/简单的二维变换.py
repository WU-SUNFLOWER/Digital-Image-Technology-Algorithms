import cv2
import numpy as np
import math

def angle2radian(angle):
    return angle / 180 * math.pi

def transform_image(image, matrix):
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

def translate_image(image, delta):
    delta_x, delta_y = delta
    matrix = np.float32([
        [1, 0, delta_x],
        [0, 1, delta_y]
    ])
    return transform_image(image, matrix)
    
def rotate_image(image, angle, is_clockwise=True, origin=(0, 0)):
    x0, y0 = origin
    theta = angle2radian(angle)
    k = -1 if is_clockwise else 1 
    matrix0 = np.float32([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ])
    matrix1 = np.float32([
        [math.cos(theta), k * math.sin(theta), 0],
        [-k * math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])
    matrix2 = np.float32([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])
    matrix = np.matmul(matrix2, np.matmul(matrix1, matrix0))
    return transform_image(image, matrix[:-1])

def horiz_flip_image(image):
    height, width, _ = image.shape
    matrix = np.float32([
        [-1, 0, width - 1],
        [0, 1, 0]
    ])
    return transform_image(image, matrix)

def vert_flip_image(image):
    height, width, _ = image.shape
    matrix = np.float32([
        [1, 0, 0],
        [0, -1, height - 1]
    ])
    return transform_image(image, matrix)

# 加载图像
image = cv2.imread('test.jpg')
height, width, _ = image.shape
center = (width / 2, height / 2)

# 应用变换
image_translated = translate_image(image, (-20, -20))

image_rotated_default = rotate_image(image, 15)
image_rotated_clockwise = rotate_image(image, 15, origin=center)
image_rotated_counter_clockwise = \
    rotate_image(image, 15, is_clockwise=False, origin=center)

image_horiz_filpped = horiz_flip_image(image)
image_vert_filpped = vert_flip_image(image)

# 保存文件
image_combine = np.hstack([
    image_translated, 
    image_rotated_default, 
    image_rotated_clockwise,
    image_rotated_counter_clockwise,
    image_horiz_filpped,
    image_vert_filpped
])
cv2.imwrite("result.jpg", image_combine)