#Imports
from unittest import result
from keras.datasets import imdb
from keras.preprocessing import sequence, text
from keras.utils import pad_sequences
import tensorflow as tf
import os
import numpy as np

#Base values
vocab_size = 88584
maxlen = 250
batch_size = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = vocab_size)

#Lets look at one review
train_data[0]

#Parse data
train_data = pad_sequences(train_data, maxlen)
test_data = pad_sequences(test_data, maxlen)
print(train_data[1])

#Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Summary
model.summary()

#Training
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

#Evaluate model
results = model.evaluate(test_data, test_labels)
print(results)

#Making predictions
word_index = imdb.get_word_index()

def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return pad_sequences([tokens], maxlen)[0]

text = 'that movie was just amazing, so amazing'
encoded = encode_text(text)
print(encoded)

#While were at it lets make a decode function
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]

print(decode_integers(encoded))

#now time to make a prediction
def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

positive_review = "That movie was so awesome!"
predict(positive_review)

negative_review = "That movie sucked"
predict(negative_review)
