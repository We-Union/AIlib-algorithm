import cv2 as cv
import numpy as np

def sift_matching(imgs, feature="SIFT", match_rule="brute", k=3, show_lines=30):
    """
    AKAZE_create
    KAZE_create
    MSER_create
    ORB_create
    SIFT_create
    """
    method = "{}_create".format(feature.upper())
    model = getattr(cv, method)()
    
    img1 = imgs[0]
    img2 = imgs[1]
    key_p1, features1 = model.detectAndCompute(img1, None)
    key_p2, features2 = model.detectAndCompute(img2, None)

    if match_rule == "brute":
        bf = cv.BFMatcher(crossCheck=True)
        match_result = bf.match(features1, features2)

    elif match_rule == "knn":
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        match_result = bf.match(features1, features2, k=k)

    match_result = sorted(match_result, key=lambda x : x.distance)
    result_img = cv.drawMatches(img1.copy(), key_p1, img2, key_p2, match_result[:show_lines], None, flags=2)    

    return result_img