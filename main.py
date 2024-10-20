from twitter_client import TwitterClient
from ai_classifier import AIClassifier

if __name__ == "__main__":
    consumer_key = 'YOUR_CONSUMER_KEY'
    consumer_secret = 'YOUR_CONSUMER_SECRET'
    access_token = 'YOUR_ACCESS_TOKEN'
    access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'
    
    twitter_client = TwitterClient(consumer_key, consumer_secret, access_token, access_token_secret)
    tweets = twitter_client.get_tweets('username', count=100)

    classifier = AIClassifier()
    train_data = ['AI-generated text example 1', 'Human-written text example 1', 'AI-generated text example 2']
    labels = [1, 0, 1]

    classifier.train(train_data, labels, epochs=10)

    for tweet in tweets:
        ai_percentage = classifier.predict(tweet)
        print(f'Tweet: {tweet}\nAI Probability: {ai_percentage:.2f}%\n')

    evaluation_data = ['AI-generated text example 3', 'Human-written text example 2']
    evaluation_labels = [1, 0]
    classifier.evaluate(evaluation_data, evaluation_labels)
