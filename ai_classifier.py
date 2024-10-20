from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_preprocessor import DataPreprocessor

class AIClassifier:
    def __init__(self, max_words=10000, max_length=100):
        self.data_preprocessor = DataPreprocessor(max_words, max_length)
        self.model = self.build_model(max_words, max_length)

    def build_model(self, max_words, max_length):
        model = Sequential()
        model.add(Embedding(max_words, 128, input_length=max_length))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=10, batch_size=32):
        self.data_preprocessor.fit(X)
        X_pad = self.data_preprocessor.transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    def predict(self, text):
        seq = self.data_preprocessor.transform([text])
        return self.model.predict(seq)[0][0] * 100

    def evaluate(self, X, y):
        X_pad = self.data_preprocessor.transform(X)
        y_pred = (self.model.predict(X_pad) > 0.5).astype("int32")
        print(classification_report(y, y_pred))
