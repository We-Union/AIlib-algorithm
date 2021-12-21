import sys
import os
import unittest

sys.path.append(os.path.abspath("."))

from main.algorithm.NLP import kanji_cut
from main.algorithm.NLP import topic_classifier
from main.algorithm.NLP import detect_mood
from main.algorithm.NLP import en2zh, zh2en


class TestWeb(unittest.TestCase):
    def test_kanji_cut(self):
        text = "李亚宁永远的神"
        spliter = ","
        _, result = kanji_cut(text, spliter=spliter, model_path="model/py_cut.pth")
        with open("text/test1.txt", "w", encoding="utf-8") as fp:
            fp.write(result)

        self.assertEqual(text, result.replace(",", ""), "分词后的文本与源文本不符合")

    def test_topic_classifier(self):
        text = "午间时分，AMD宣布新一代的服务器级CPU投入量产，这势必会让整个芯片市场的开始转移核心竞争力"
        _, result = topic_classifier(text)
        with open("text/test1.txt", "w", encoding="utf-8") as fp:
            fp.write(result)

        self.assertEqual(result, "科技", "话题分类错误")    

    def test_zh2en(self):
        text = "你好，世界"
        _, result = zh2en(text)

        with open("text/test1.txt", "w", encoding="utf-8") as fp:
            fp.write(result)

    def test_en2zh(self):
        text = "hello world"
        _, result = en2zh(text)

        with open("text/test1.txt", "w", encoding="utf-8") as fp:
            fp.write(result)

    def test_detect_mood(self):
        text = "hello world, I am glad to see you"
        _, text = en2zh(text)        
        _, result = detect_mood(text)
        
        with open("text/test1.txt", "w", encoding="utf-8") as fp:
            fp.write(text)
            fp.write("\n")
            fp.write(result)

if __name__ == "__main__":
    unittest.main()
