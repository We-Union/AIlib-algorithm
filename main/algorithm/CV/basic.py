import cv2 as cv

def equalize_hist(img, local=False, clipLimit=4.0, tileGridSize=4):
    if local:
        clane = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
        f = clane.apply
    else:
        f = cv.equalizeHist
    if len(img.shape) == 3:
        r, g, b = cv.split(img)
        r = f(r)
        g = f(g)
        b = f(b)
        result = cv.merge([r, g, b])
        result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    else:
        result = f(img)
    return result, "分析成功"

def OSTU_split(img, blur_size=3, blur_std=1, reverse=False):
    if blur_size % 2 == 0:
        blur_size += 1
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, ksize=(blur_size, blur_size), sigmaX=blur_std)
    _, binary = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    if reverse:
        binary = 255 - binary
    return binary, "分析成功"