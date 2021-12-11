import cv2 as cv
from skimage import io
import numpy as np

def resize(img : np.ndarray, height=None, width=None) -> np.ndarray:
    if height is None and width is None:
        raise ValueError("not None at the same time")
    if height is not None and width is not None:
        raise ValueError("not not None at the same time")
    h, w = img.shape[0], img.shape[1]
    if height:
        width = int(w / h * height)
    else:
        height = int(h / w * width)
    target_img = cv.resize(img, dsize=(width, height))
    return target_img

def show_image(img, winname = 'Default', height = None, width = None, format="bgr"):
    if format.lower() == "rgb":
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            
    cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)
    if height or width:
        img = resize(img, height, width)
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def url_imread(url : str, out_format : str = "bgr") -> np.ndarray:
    img = io.imread(url)
    if out_format.lower() == "rgb":
        return img
    elif out_format.lower() == "bgr":
        return cv.cvtColor(img, cv.COLOR_RGB2BGR)

if __name__ == "__main__":
    url = "https://tse4-mm.cn.bing.net/th/id/OIP-C.FsrjjaLX9dlUa89zxzHbyQHaJ3?pid=ImgDet&rs=1"
    img = url_imread(url)
    show_image(img)