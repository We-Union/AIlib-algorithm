import sys
import os  
sys.path.append(os.path.abspath("."))
# from pprint import  pprint

# pprint(sys.path)
# print(__file__)

from main.algorithm.CV import scanning
from main.algorithm.CV import transform_to_painting
from main.algorithm.CV import show_image
from main.algorithm.CV import sift_matching
from main.algorithm.CV import reconstruct
from main.algorithm.CV import detect_face
from main.algorithm.CV import stitching
from main.algorithm.CV import ocr_val, ocr_print
from main.algorithm.CV import equalizeHist, OSTU_split

import cv2 as cv
import numpy as np

def u_transform_to_painting():
    img = cv.imread("../image/test2.jpg")
    img = transform_to_painting(img, depth=200, blur=True, blur_size=5, blur_std=2, denoise=True, denoise_size=5)
    show_image(img)

def u_scanning():
    img = cv.imread("image/card.jpg")
    img = scanning(img, height=600)
    if isinstance(img, int):
        print("error code 5")
    else:
        show_image(img)

def u_sift_matching():
    img1 = cv.imread("image/book.jpg")
    img2 = cv.imread("image/desktop.jpg")
    img = sift_matching([img1, img2], feature="akaze")
    show_image(img, height=600)

def u_hrr():
    img = cv.imread("image/test2.jpg")
    result = reconstruct(img, outscale=1)
    show_image(np.concatenate([img, result], axis=1), height=600)

def u_face_detect():
    img = cv.imread("image/lena.png")
    result = detect_face(img, method="haar")
    show_image(result, format='rgb')

def u_stitching():
    img1 = cv.imread("image/img1.jpg")
    img2 = cv.imread("image/img2.jpg")
    
    # show_image(np.concatenate([img1, img2], axis=1), width=1200)

    result = stitching(img1, img2)
    show_image(result, width=1500, format='rgb')

def u_test_ocr():
    img = cv.imread("image/ocr2.jpg")
    text = ocr_val(img)
    print(text)

def u_equalizeHist():
    img = cv.imread("image/lena.png")
    result = equalizeHist(img, local=False)
    show_image(result, format='rgb')

def u_OSTU():
    img = cv.imread("image/lena.png")
    result = OSTU_split(img, reverse=True)
    show_image(result)


# u_hrr()
# u_face_detect()
# u_stitching()
# u_test_ocr()

u_OSTU()