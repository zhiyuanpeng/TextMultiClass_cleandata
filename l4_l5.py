import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input

import pandas as pd
import re

from numpy import asarray
from numpy import zeros
import numpy as np

import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# read the train data
toxic_comments = pd.read_csv("data/unprocess/train.csv")
filter = toxic_comments["comment_text"] != ""
toxic_comments = toxic_comments[filter]
toxic_comments = toxic_comments.dropna()
# we only train a binary classifier for lable 1
# [l0, l1, l2, l3, l4, l5]
# ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = np.array(toxic_comments[["insult"]])*np.array(toxic_comments[["identity_hate"]])


# clear text
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


# write the list
def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(float(line_value)) + "\n")


def y_write_round(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            if float(line_value) > 0.5:
                f.write(str(1) + "\n")
            else:
                f.write(str(0) + "\n")


X = []
sentences = list(toxic_comments["comment_text"])
for sen in sentences:
    X.append(preprocess_text(sen))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# write the X_train, X_test, y_train, y_test
x_write(X_train, "data/store/l4_l5_X_train.txt")
x_write(X_test, "data/store/l4_l5_X_test.txt")
y_write(y_train, "data/store/l4_l5_y_train.txt")
y_write(y_test, "data/store/l4_l5_y_test.txt")
#
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()

glove_file = open('../glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(1, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
predict_list = model.predict(X_test, batch_size=128, verbose=1)
y_write(predict_list, "data/store/l4_l5_predict.txt")
y_write_round(predict_list, "data/store/l4_l5_predict_round.txt")
right_sum = 0
for index in range(len(predict_list)):
    if float(predict_list[index]) > 0.5:
        y_predict = 1
    else:
        y_predict = 0
    if y_predict == int(y_test[index]):
        right_sum += 1
accuracy = (right_sum/len(predict_list))*100

# save the result
with open("data/store/l4_l5_result.txt", "a+") as r:
    r.write("self accuracy is " + str(accuracy) + "\n")
    r.write("score is " + str(score[0]) + "\n")
    r.write("accuracy is " + str(score[1]) + "\n")

model.save("models/l4_l5.h5")
