# from github susanli2016

from keras.datasets import imdb

vocabulary_size = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)

from keras.preprocessing import sequence

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_valid, y_valid = X_train[:64], y_train[:64]
X_train2, y_train2 = X_train[64:], y_train[64:]

model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=64, epochs=5)
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy: ', scores[1])
