from tensorflow_core.python.keras.datasets import imdb
from tensorflow_core.python.keras.preprocessing import sequence
import numpy as np
max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


from tensorflow_core.python.keras import models
from tensorflow_core.python.keras import layers

network = models.Sequential()
network.add(layers.Embedding(max_features, 128))
network.add(layers.Conv1D(256,3,padding='valid',activation='relu',strides=1))
network.add(layers.MaxPooling1D())
network.add(layers.Bidirectional(layers.LSTM(128)))
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1))
network.add(layers.Activation('sigmoid'))

#make the training data 80% and testing 20%
input_train = np.concatenate((input_train, input_test[:15000]))
input_test = input_test[15000:]
y_train = np.concatenate((y_train, y_test[:15000]))
y_test = y_test[15000:]

network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
his = network.fit(input_train, y_train,
                    epochs=4,
                    batch_size=64,
                    validation_split=0.025)


results = network.evaluate(input_test, y_test)
print(results)

print("Done!")