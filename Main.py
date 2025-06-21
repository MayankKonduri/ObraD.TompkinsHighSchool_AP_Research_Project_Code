from twitter_scraper import TwitterScraper
from model import AITweetClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
scraper = TwitterScraper(BEARER_TOKEN)
tweets_df = scraper.fetch_tweets("AI", max_results=50)

# Simulated labels (replace with real data)
tweets_df['label'] = [0 if i % 2 == 0 else 1 for i in range(len(tweets_df))]  # 0 = Human, 1 = AI

texts = tweets_df['text'].tolist()
labels = tweets_df['label'].tolist()

# K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kfold.split(texts, labels)):
    print(f"\nFold {fold + 1}")
    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    classifier = AITweetClassifier()
    classifier.train(X_train, y_train, epochs=3)

    predictions = classifier.predict(X_test)
    predicted_labels = [1 if p > 0.5 else 0 for p in predictions]

    print(classification_report(y_test, predicted_labels))

    # Confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - Fold {fold + 1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Fold {fold + 1}')
    plt.legend()
    plt.show()
