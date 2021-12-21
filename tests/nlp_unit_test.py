import sys
import os
sys.path.append(os.path.abspath("."))

from main.algorithm.NLP import detect_mood
from main.algorithm.NLP import en2zh, zh2en

def u_detect_mood():
    text = "hello world, I am glad to see you"
    _, text = en2zh(text)    

    print(text)
    
    _, result = detect_mood(text)

    print(result)

u_detect_mood()