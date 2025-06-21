import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

class AITweetClassifier:
    def __init__(self, vocab_size=10000, max_length=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=64, input_length=self.max_length),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def preprocess(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded

    def train(self, texts, labels, epochs=5):
        self.tokenizer.fit_on_texts(texts)
        X = self.preprocess(texts)
        y = np.array(labels)
        self.model.fit(X, y, epochs=epochs)

    def predict(self, texts):
        X = self.preprocess(texts)
        return self.model.predict(X)

    def evaluate(self, texts, labels):
        X = self.preprocess(texts)
        y = np.array(labels)
        return self.model.evaluate(X, y)

    def save(self, model_path="model.h5", tokenizer_path="tokenizer.pkl"):
        self.model.save(model_path)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load(self, model_path="model.h5", tokenizer_path="tokenizer.pkl"):
        self.model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
