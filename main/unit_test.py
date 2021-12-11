import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from algorithm.CV import transform_to_painting
from algorithm.CV.utils import show_image, url_imread
from my_io import upload_sm, get_sm_token

def test_sm(img):
    global_token = get_sm_token()
    img = cv.imread("image/test1.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    url = upload_sm(global_token, img)
    print(url)

    img = url_imread(url)
    show_image(img, format="rgb")

def test_painting(url):
    img = url_imread(url)
    show_image(img)
    img = transform_to_painting(img, depth=250)
    show_image(img)    
    return img

url = "https://s2.loli.net/2021/12/11/bBwO4Y1pfqloaP6.jpg"
test_painting(url)