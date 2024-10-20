import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DataPreprocessor:
    def __init__(self, max_words=10000, max_length=100):
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=self.max_words)

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_length)
