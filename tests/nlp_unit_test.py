import sys
import os
sys.path.append(os.path.abspath("."))

from main.algorithm.NLP import detect_mood, topic_classifier
from main.algorithm.NLP import en2zh, zh2en

def u_topic_classifier():
    text = "午间时分，AMD宣布新一代的服务器级CPU投入量产，这势必会让整个芯片市场的开始转移核心竞争力"
    _, result = topic_classifier(text, out_dict_str=True)

    print(result)

def u_detect_mood():
    text = "hello world, I am glad to see you"
    # _, text = en2zh(text)    

    # print(text)

    _, result = detect_mood(text, out_dict_str=True)

    print(result)

# u_detect_mood()
u_topic_classifier()