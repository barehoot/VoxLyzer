# text_sentiment.py
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def text_sent(content):
    sia = SentimentIntensityAnalyzer()
    # Your sentiment analysis code here
