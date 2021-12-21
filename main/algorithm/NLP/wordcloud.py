import wordcloud
import jieba.posseg as pseg
import os
import numpy as np

def generate_wordcloud(text, lag='zh', punc=True, width=600, height=200, stop_words=None, max_words=200, background_color="white"):
    if len(text) == 0:
        return 6011, None
    if stop_words is not None:
        stop_words = set(stop_words)
    font_path="msyh.ttc"
    wc = wordcloud.WordCloud(width=width, height=height, font_path=font_path, stopwords=stop_words, max_words=max_words, background_color=background_color)
    
    if   lag == 'zh':
        if punc:
            words = []
            filter_list = ['x', 'df', 'ud', 'ug', 'uj', 'ul', 'uv', 'uz', 'y', 'u', 'rz', 'p']
            for w in pseg.cut(text):
                if w.flag in filter_list:
                    continue
                words.append(w.word)
            if len(words) == 0:
                return 6011, None
            text = " ".join(words)

        wc.generate(text)
        return np.array(wc.to_image()), "分析成功"

    elif lag == 'en':
        if punc:
            words = []
            filter_list = ['x']
            for w in pseg.cut(text):
                if w.flag in filter_list:
                    continue
                words.append(w.word)
            if len(words) == 0:
                return 6011, None
            text = " ".join(words)

        wc.generate(text)
        return np.array(wc.to_image()), "分析成功"

    else:
        return 6010, None
    
