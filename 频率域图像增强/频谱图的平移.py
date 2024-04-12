import cv2
import numpy as np
import math
import datetime

def Discrete_Fourier_Transform(source, height, width, is_positive=True):
    # 初始化保存结果的矩阵
    F = np.zeros((height, width), dtype=complex)
    
    coef = 2 * math.pi * (-1 if is_positive else 1)

    # 创建辅助计算的列向量↑↓
    vector_col = np.arange(height)
    # 创建辅助计算的行向量←→
    vector_row = np.arange(width)
    u_matrix = np.exp(coef * (np.outer(vector_row, vector_row) / width) * 1j)
    # 为了利用NumPy的广播机制，别忘了要把行向量转成列向量
    v_matrix = np.exp(coef * (np.outer(vector_col, vector_col) / height) * 1j).reshape(height, height, 1)
    # 进行变换
    for v in range(height):
        for u in range(width):
            # 计算点积
            matrix = u_matrix[u] * v_matrix[v]
            F[v, u] = np.dot(source.ravel(), matrix.ravel())
    # 除以常系数,保证变换后的信号能正确地反映原始信号的幅度
    F /= math.sqrt(height * width)
    # 如果是傅里叶反变换，返回前还要进行一些处理
    if not is_positive: 
        F = F.real.astype(np.uint8).clip(0, 255)
    return F

image = cv2.imread("checker.jpg", cv2.IMREAD_GRAYSCALE)
height, width = image.shape
start_time = datetime.datetime.now()

# 对原图像在空间域进行预处理
u = np.arange(width)
v = np.arange(height).reshape(height, 1)
image = image * np.power(-1, u + v)

# 计算傅里叶变换结果
F = Discrete_Fourier_Transform(image, height, width)

end_time = datetime.datetime.now()
print("消耗时间: ", end_time - start_time)

# 显示频谱图
graph = np.abs(F).astype(np.uint8).clip(0, 255)
cv2.imshow('graph', graph)

cv2.waitKey()
cv2.destroyAllWindows()