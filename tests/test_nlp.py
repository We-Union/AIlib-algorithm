import sys
import os
import unittest

sys.path.append(os.path.abspath("."))

from main.algorithm.CV  import show_image
from main.algorithm.NLP import kanji_cut
from main.algorithm.NLP import topic_classifier
from main.algorithm.NLP import detect_mood
from main.algorithm.NLP import en2zh, zh2en
from main.algorithm.NLP import generate_wordcloud
from main.algorithm.NLP import visual_wordvec
from main.algorithm.NLP import talk_to_chatbot


class TestWeb(unittest.TestCase):
    def test_kanji_cut(self):
        text = "中国男子吊环项目始终有着深厚的冠军底蕴。“体操王子”李宁在1984年的洛杉矶奥运会上斩获的金牌就来自于吊环项目；2008年北京奥运会，陈一冰也斩获了吊环金牌；刘洋在经历漫长而枯燥的训练和无数次与伤病抗争后，在东京奥运会上成功加冕“吊环王”。"
        spliter = ","
        _, result = kanji_cut(text, spliter=spliter, model_path="../model/py_cut.pth")
        with open("../text/test1.txt", "w", encoding="utf-8") as fp:
            fp.write(result)
        print(result)
        # self.assertEqual(text, result.replace(",", ""), "分词后的文本与源文本不符合")

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
        _, result = detect_mood(text, out_dict_str=False)
        
        self.assertEqual(result, "正面心情", "情感分类错误")   

    def test_wordcloud(self):
        with open("text/text2.txt", "r", encoding='utf-8') as fp:
            text = fp.read().replace('\n', ' ') 
        img, _ = generate_wordcloud(text, lag='zh', background_color='white')
        show_image(img, format='rgb', width=1000)

    def test_visual_wordvec(self):
        with open("text/text3.txt", "r", encoding='utf-8') as fp:
            word_list = fp.read().replace("\n", " ")
        img, _ = visual_wordvec(word_list)
        show_image(img)
    
    def test_chatbot(self):
        text = "hello, what is your name ?"
        _, response = talk_to_chatbot(text)
        

if __name__ == "__main__":
    unittest.main()
