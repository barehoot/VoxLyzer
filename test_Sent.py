import pytest
from your_main_script import text_sent

@pytest.fixture
def positive_content():
    return "I love this tool. It's great!"

@pytest.fixture
def negative_content():
    return "This is terrible. I hate it!"

@pytest.fixture
def neutral_content():
    return "It's okay, not too good or bad."

def test_text_sentiment_positive(positive_content):
    sentiment = text_sent(positive_content)
    assert sentiment['compound'] > 0

def test_text_sentiment_negative(negative_content):
    sentiment = text_sent(negative_content)
    assert sentiment['compound'] < 0

def test_text_sentiment_neutral(neutral_content):
    sentiment = text_sent(neutral_content)
    assert sentiment['compound'] == 0
