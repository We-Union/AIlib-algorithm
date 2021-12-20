import sys
import os
import unittest

sys.path.append(os.path.abspath("."))

from main.algorithm.NLP import kanji_cut


class TestWeb(unittest.TestCase):
    def test_kanji_cut(self):
        text = "黄哲龙永远的神"
        spliter = ","
        result = kanji_cut(text, spliter=spliter, model_path="../model/py_cut.pth")
        print(result)
        self.assertEqual(text, result.replace(",", ""), "分词后的文本与源文本不符合")


if __name__ == "__main__":
    unittest.main()
