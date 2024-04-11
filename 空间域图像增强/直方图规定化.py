import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw(picture, title, data):
    x_coords = range(len(data))
    picture.bar(x_coords, data)
    picture.set_title(title)
    picture.set_xlabel('Gray Value')
    picture.set_ylabel('Frequency')
    picture.set_xticks(x_coords, labels=x_coords, rotation=90)

# 定义一个函数来计算累计分布函数（CDF）
def calc_cdf(hist):
    count_pixels = hist.sum() # 像素点总数
    frequencies = hist.ravel() / count_pixels # 计算不同灰度级在图中出现的频率
    cumulative_sum = frequencies.cumsum() # 计算各灰度级对应的CDF
    return cumulative_sum

# 加载图像
source_img = cv2.imread('source_image.jpg', cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread('reference_image.jpg', cv2.IMREAD_GRAYSCALE)

# 计算两幅图像的直方图
source_hist = cv2.calcHist([source_img], [0], None, [256], [0, 256])
reference_hist = cv2.calcHist([reference_img], [0], None, [256], [0, 256])

# 计算CDF
source_cdf = calc_cdf(source_hist)
reference_cdf = calc_cdf(reference_hist)

# 构建映射函数
M = np.zeros((256,),dtype=np.uint8)
for a in range(0, 256):
    # 查找与a最接近的CDF值
    # 这里运用了NumPy Broadcasting技巧
    j = np.abs(source_cdf[a] - reference_cdf).argmin()
    M[a] = j

# 应用映射函数
# 这里运用了NumPy Fancy Indexing技巧
result_img = M[source_img]

# 显示结果
fig, (picture1, picture2, picture3) = plt.subplots(1, 3, figsize=(25, 6))

source_hist = cv2.calcHist([source_img], [0], None, [32], [0, 256])
draw(picture1, "Source Image", source_hist.ravel())

reference_hist = cv2.calcHist([reference_img], [0], None, [32], [0, 256])
draw(picture2, "Reference Image", reference_hist.ravel())

result_hist = cv2.calcHist([result_img], [0], None, [32], [0, 256])
draw(picture3, "Result Image", result_hist.ravel())

plt.tight_layout()
plt.show()