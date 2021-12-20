import sys
import os  
sys.path.append(os.path.abspath("."))

from main.algorithm.NLP import kanji_cut

def u_kanji_cut():
    text = "黄哲龙永远的神"
    result = kanji_cut(text)
    print(result)

u_kanji_cut()