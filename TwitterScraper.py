import tweepy
import pandas as pd
import os
import re
from langdetect import detect

class TwitterScraper:
    def __init__(self, bearer_token):
        self.client = tweepy.Client(bearer_token=bearer_token)

    def clean_text(self, text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def fetch_tweets(self, query, max_results=100):
        tweets = self.client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=['created_at', 'author_id', 'text'])
        tweet_data = []
        seen_texts = set()
        for tweet in tweets.data:
            cleaned_text = self.clean_text(tweet.text)
            if cleaned_text and cleaned_text not in seen_texts and detect(cleaned_text) == 'en':
                seen_texts.add(cleaned_text)
                tweet_data.append({
                    'author_id': tweet.author_id,
                    'created_at': tweet.created_at,
                    'text': cleaned_text
                })
        return pd.DataFrame(tweet_data)
