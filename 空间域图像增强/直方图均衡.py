import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像（以灰度图的形式）
img = cv2.imread('./lenna.jpg', cv2.IMREAD_GRAYSCALE)

# 计算不同灰度值出现的频率
height, width = img.shape
frequencies = np.zeros(256, dtype=np.int32)
for i in range(height):
    for j in range(width):
        frequencies[img[i][j]] += 1

# 计算映射函数
cdf = frequencies.cumsum() / (height * width)
gray_value_map = np.zeros(256, dtype=np.uint8)
for i in range(0, 256):
    gray_value_map[i] = min(round(cdf[i] * 255), 255)
    
# 应用直方图均衡化
equ_img = gray_value_map[img]

# 显示原始图像和均衡化后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image', equ_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 为了更直观地比较，我们还可以绘制它们的直方图
plt.figure()
plt.hist(img.ravel(), 256, [0, 256], color='r')
plt.hist(equ_img.ravel(), 256, [0, 256], color='g')
plt.title('Histogram for original and equalized image')
plt.show()