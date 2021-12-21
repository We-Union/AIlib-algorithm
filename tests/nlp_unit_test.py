import sys
import os

sys.path.append(os.path.abspath("."))

from main.algorithm.CV  import show_image
from main.algorithm.NLP import detect_mood, topic_classifier
from main.algorithm.NLP import en2zh, zh2en, generate_wordcloud
from main.algorithm.NLP import visual_wordvec

import jieba.posseg as pseg

def u_topic_classifier():
    text = "午间时分，AMD宣布新一代的服务器级CPU投入量产，这势必会让整个芯片市场的开始转移核心竞争力"
    _, result = topic_classifier(text, out_dict_str=True)

    print(result)

def u_detect_mood():
    text = "what a bad weather"

    _, result = detect_mood(text, out_dict_str=True)

    print(result)

def u_wordcloud():
    # text = "this is a nice weather"
    with open("text/text2.txt", "r", encoding='utf-8') as fp:
        text = fp.read().replace('\n', ' ') 
    img, _ = generate_wordcloud(text, lag='zh', background_color='white')
    show_image(img, format='rgb', width=1000)

def u_visual_wordvec():
    with open("text/text3.txt", "r", encoding="utf-8") as fp:
        vis_word = fp.read().replace("\n", " ")
    img, _ = visual_wordvec(vis_word, decomposition_method="PCA")
    show_image(img, width=1200)
    

# u_detect_mood()
# u_topic_classifier()

u_visual_wordvec()