import tweepy

class TwitterClient:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)

    def get_tweets(self, username, count=100):
        tweets = self.api.user_timeline(screen_name=username, count=count, tweet_mode='extended')
        return [tweet.full_text for tweet in tweets]
