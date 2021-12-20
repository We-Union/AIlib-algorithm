from typing import Tuple
import cv2 as cv
import numpy as np
from main.algorithm.CV.utils import resize, show_image

def point_sorted(points : np.ndarray) -> Tuple[np.ndarray]:
    """
        sort as:
        tl  tr
        bl  br
    """

    res = sorted(points, key=lambda p : p[1])
    tops = res[:2]
    bottoms = res[2:]

    if tops[0][0] > tops[1][0]:
        tl = tops[1]
        tr = tops[0]
    else:
        tl = tops[0]
        tr = tops[1]
    
    if bottoms[0][0] > bottoms[1][0]:
        bl = bottoms[1]
        br = bottoms[0]
    else:
        bl = bottoms[0]
        br = bottoms[1]
    
    return tl, tr, bl, br

def transform(img, points):
    tl, tr, bl, br = point_sorted(points)

    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    src_points = np.array([tl, tr, br, bl], dtype="float32")
    dst_points = np.array([
        [0,                  0],
        [width - 1,          0],
        [width - 1, height - 1],
        [0,         height - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(src_points, dst_points)
    transformed = cv.warpPerspective(img, M, (width, height))
    return transformed

def scanning(img, height=500):
    img = resize(img, height=height)
    
    # canny detection
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edge = cv.Canny(gray, 75, 100)

    # detect contour
    contours, _ = cv.findContours(edge.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contours = contours[:5]

    # s_img = img.copy()
    # cv.drawContours(s_img, contours, -1, (0, 255, 0), 3)

    screenCnt = None
    for contour in contours:
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        return 6005
    
    # cv.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    # showImage(img)
    transformed = transform(img, screenCnt.reshape(4, 2))
    # showImage(transformed)
    return cv.cvtColor(transformed, cv.COLOR_BGR2RGB),"分析成功"