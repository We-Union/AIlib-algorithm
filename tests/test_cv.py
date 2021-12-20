import unittest
from main.algorithm.CV import transform_to_painting
from main.algorithm.CV.utils import show_image, url_imread
import sys
import os
sys.path.append(os.path.abspath("."))

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


class TestPainting(unittest.TestCase):
    def test_painting(self):
        url = "https://s2.loli.net/2021/12/11/bBwO4Y1pfqloaP6.jpg"
        img = url_imread(url)
        show_image(img)
        img = transform_to_painting(img, depth=250)
        show_image(img)
        return img

    def test_transform_to_painting(self):
        img = cv.imread("../image/test2.jpg")
        img = transform_to_painting(img, depth=200, blur=True, blur_size=5, blur_std=2, denoise=True, denoise_size=5)
        show_image(img)

    def test_scanning(self):
        img = cv.imread("image/card.jpg")
        img = scanning(img, height=600)
        if isinstance(img, int):
            print("error code 5")
        else:
            show_image(img)

    def test_sift_matching(self):
        img1 = cv.imread("image/book.jpg")
        img2 = cv.imread("image/desktop.jpg")
        img = sift_matching([img1, img2], feature="akaze")
        show_image(img, height=600)

    def test_hrr(self):
        img = cv.imread("image/test2.jpg")
        result = reconstruct(img, outscale=1)
        show_image(np.concatenate([img, result], axis=1), height=600)

    def test_face_detect(self):
        img = cv.imread("image/lena.png")
        result = detect_face(img, method="haar")
        show_image(result, format='rgb')

    def test_stitching(self):
        img1 = cv.imread("../image/img1.jpg")
        img2 = cv.imread("../image/img2.jpg")

        # show_image(np.concatenate([img1, img2], axis=1), width=1200)

        result = stitching([img1, img2])
        show_image(result, width=1500, format='rgb')

    def test_ocr(self):
        img = cv.imread("image/ocr2.jpg")
        text = ocr_val(img)
        print(text)

    def test_equalizeHist(self):
        img = cv.imread("image/lena.png")
        result = equalizeHist(img, local=False)
        show_image(result, format='rgb')

    def test_OSTU(self):
        img = cv.imread("image/lena.png")
        result = OSTU_split(img, reverse=True)
        show_image(result)

if __name__ == "__main__":
    unittest.main()
