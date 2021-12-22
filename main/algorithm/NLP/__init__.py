__version__ = "0.0.1"

from main.algorithm.NLP.KanjiCut import kanji_cut
from main.algorithm.NLP.topic_classifier import topic_classifier
from main.algorithm.NLP.mood_detect import detect_mood
from main.algorithm.NLP.translate import en2zh, zh2en
from main.algorithm.NLP.wordcloud import generate_wordcloud
from main.algorithm.NLP.wordvec_visual import visual_wordvec
from main.algorithm.NLP.chatbot import talk_to_chatbot

from main.algorithm.NLP.utils import txtread
