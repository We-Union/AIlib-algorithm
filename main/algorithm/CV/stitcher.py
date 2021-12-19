import cv2 as cv
import numpy as np
from main.algorithm.CV.utils import resize, show_image

class Stitcher(object):
    def __init__(self, feature) -> None:
        method = "{}_create".format(feature.upper())
        self.descriptor = getattr(cv, method)()
        self.matcher = cv.BFMatcher()
    
    def detect(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kps, fs = self.descriptor.detectAndCompute(gray, None)
        kps = np.array([kp.pt for kp in kps], dtype="float32")
        return kps, fs
    
    def drawMatch(self, img1, img2, showPst : int = 50):
        gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        kps1, fs1 = self.descriptor.detectAndCompute(gray, None)
        gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        kps2, fs2 = self.descriptor.detectAndCompute(gray, None)
        match_result = self.matcher.match(fs1, fs2)
        result = cv.drawMatches(img1, kps1, img2, kps2, match_result[:min(showPst, len(match_result))], None)
        return result

    def stitch(self, img1, img2, ratio = 0.75, reproThreshold = 0.4):  
        img1 = resize(img1, width=1000)
        img2 = resize(img2, width=1000)

        kps1, fs1 = self.detect(img1)
        kps2, fs2 = self.detect(img2)

        # get transform
        rawMathes = self.matcher.knnMatch(fs1, fs2, 2)
        matches = []
        for m in rawMathes:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # m[0].trainIdx and m[0].queryIdx are index of m[0] in fs1 and fs2
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
        H = None
        # the transform needs at least 8 values to solve, which is 4 points
        if len(matches) > 4:
            pts_img1 = np.array([kps1[i] for _, i in matches], dtype='float32')
            pts_img2 = np.array([kps2[i] for i, _ in matches], dtype='float32')

            H, status = cv.findHomography(pts_img1, pts_img2, cv.RANSAC, reproThreshold)

        if H is None:
            return 6005

        result = cv.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        result[:img2.shape[0], :img2.shape[1]] = img2
        return result

def stitching(img1, img2, feature="akaze", ratio=0.75, reproThreshold=0.4):
    stitcher = Stitcher(feature=feature)
    result = stitcher.stitch(img2, img1, ratio, reproThreshold)
    return cv.cvtColor(result, cv.COLOR_BGR2RGB)