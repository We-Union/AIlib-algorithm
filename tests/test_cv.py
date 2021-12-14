import unittest
from main.algorithm.CV import transform_to_painting
from main.algorithm.CV.utils import show_image, url_imread



class TestPainting(unittest.TestCase):
    def test_painting(self):
        url = "https://s2.loli.net/2021/12/11/bBwO4Y1pfqloaP6.jpg"
        img = url_imread(url)
        show_image(img)
        img = transform_to_painting(img, depth=250)
        show_image(img)
        return img


if __name__ == "__main__":
    unittest.main()
