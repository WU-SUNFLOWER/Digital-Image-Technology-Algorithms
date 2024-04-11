import cv2
import numpy as np
import datetime

def cubic(t):
    t = abs(t)
    if 1 < t and t < 2:
        return 4 - 8 * t + 5 * (t ** 2) - (t ** 3)
    elif t <= 1:
        return 1 - 2 * (t ** 2) + (t ** 3)
    else:
        return 0 

def my_resize(image, scale_factor):
    height, width = image.shape
    new_width = round(width * scale_factor)
    new_height = round(height * scale_factor)
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8)

    x_border = width - 1
    y_border = height - 1

    for i in range(new_height):
        for j in range(new_width):
            x = np.clip(j / scale_factor, 0, x_border)
            y = np.clip(i / scale_factor, 0, y_border)
            x_int, y_int = int(x), int(y)
            dx, dy = x - x_int, y - y_int

            y_low = max(y_int - 1, 0)
            y_high = min(y_int + 2, y_border)
            y_size = y_high - y_low + 1
            
            x_low = max(x_int - 1, 0)
            x_high = min(x_int + 2, x_border)
            x_size = x_high - x_low + 1
            
            weights_x_axis = np.empty(x_size)
            for k in range(x_size):
                weights_x_axis[k] = cubic(k - 1 - dx)
            
            weights_y_axis = np.empty(y_size)
            for k in range(y_size):
                weights_y_axis[k] = cubic(k - 1 - dy)
            
            resized_image[i, j] = np.dot(
                image[y_low:y_high+1, x_low:x_high+1].ravel(),
                np.outer(weights_y_axis, weights_x_axis).ravel()
            ).clip(0, 255)
    
    return resized_image

def main():
    image_path = 'lenna.jpg'  # 更换为你的图片路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 缩放图像
    start_time = datetime.datetime.now()
    resized_image = my_resize(image, 2)
    end_time = datetime.datetime.now()
    print("运行时间：", end_time - start_time)

    # 保存结果
    cv2.imwrite('~lenna.jpg', resized_image)

if __name__ == '__main__':
    main()