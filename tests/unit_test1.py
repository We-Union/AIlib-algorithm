from main.algorithm.CV import scanning
from main.algorithm.CV import transform_to_painting
from main.algorithm.CV import show_image
from main.algorithm.CV import sift_matching
from main.algorithm.CV import reconstruct

import cv2 as cv
import numpy as np

def u_transform_to_painting():
    img = cv.imread("../image/test2.jpg")
    img = transform_to_painting(img, depth=200, blur=True, blur_size=5, blur_std=2, denoise=True, denoise_size=5)
    show_image(img)

def u_scanning():
    img = cv.imread("../image/book.jpg")
    img = scanning(img, height=600)
    if isinstance(img, int):
        print("error code 5")
    else:
        show_image(img)

def u_sift_matching():
    img1 = cv.imread("../image/book.jpg")
    img2 = cv.imread("../image/desktop.jpg")
    img = sift_matching([img1, img2], feature="akaze")
    show_image(img, height=600)

def u_hrr():
    img = cv.imread("../image/test2.jpg")
    result = reconstruct(img, outscale=1)
    show_image(np.concatenate([img, result], axis=1), height=600)

u_hrr()