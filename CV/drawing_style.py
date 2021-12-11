from PIL import Image
import numpy as np
from CV.utils import show_image
import cv2 as cv

def transform_to_painting(img, depth=100):
    """
        depth : [0, 100]
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grad = np.gradient(img) # 取图像灰度的梯度

    grad_x, grad_y = grad # 分别取图像横纵方向灰度值的梯度值
    grad_x = grad_x * depth / 100. #将横纵灰度值的梯度值归一化
    grad_y = grad_y * depth / 100.

    A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.) #继续归一化
    uni_x = grad_x / A #x，y，z表示图像平面的单位法向量在三个轴上的投影
    uni_y = grad_y / A
    uni_z = 1 / A

    vec_el = np.pi / 2.2 #光源的俯视角度
    vec_az = np.pi / 4. #光源的方位角度
    dx = np.cos(vec_el) * np.cos(vec_az) #光源对x轴的影响因子
    dy = np.cos(vec_el) * np.sin(vec_az) #光源对y轴的影响因子
    dz = np.sin(vec_el) #光源对z轴的影响因子

    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z) #将各方向的梯度分别乘上虚拟光源对各方向的影响因子，将梯度还原成灰度
    b = b.clip(0, 255) #舍弃溢出的灰度值
    return b

