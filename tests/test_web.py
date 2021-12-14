import cv2 as cv
import unittest
from main.my_io import upload_sm, get_sm_token


class TestWeb(unittest.TestCase):
    def test_sm_upload(self):
        global_token = get_sm_token("../algorithm/config.json")
        img = cv.imread("../image/test1.jpg")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        url = upload_sm(global_token, img)
        print(url)


if __name__ == "__main__":
    unittest.main()
