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

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_predict_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value[0]) + "\n")


def y_write_round(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            if float(line_value) > 0.5:
                f.write(str(1) + "\n")
            else:
                f.write(str(0) + "\n")


def read_text(filename):
    """
    read the train text to list
    @param filename: the file name of the train.txt
    @return: a list contains all the text
    """
    text_list = []
    with open(filename, "r") as f:
        text = f.readlines()
        for line in text:
            text_list.append(line.strip("\n"))
    return text_list


def get_edge_label(node_a, node_b, file_name):
    y_a = np.loadtxt("data/processed/" + file_name, dtype=int)[:, node_a]
    y_b = np.loadtxt("data/processed/" + file_name, dtype=int)[:, node_b]
    edge_label = [y_a[i]*y_b[i] for i in range(y_b.shape[0])]
    return np.array(edge_label)


# unbalance data
X_train_ub = read_text("data/processed/X_train.txt")
y_train_ub = get_edge_label(0, 5, "y_train.txt")
# balance data
non_zero = np.sum(y_train_ub)
rate = non_zero/(len(y_train_ub) - non_zero)
X_train = []
y_train = []
np.random.seed(0)
for i in range(y_train_ub.shape[0]):
    if y_train_ub[i, ] == 1:
        X_train.append(X_train_ub[i])
        y_train.append(1)
    elif np.random.random() <= rate:
        X_train.append(X_train_ub[i])
        y_train.append(0)
y_train = np.array(y_train)
#
X_test = read_text("data/processed/X_test.txt")
y_test = get_edge_label(0, 5, "y_test.txt")
# record the train and test data
x_write(X_train, "data/store/l0_l5/l0_l5_X_train.txt")
x_write(X_test, "data/store/l0_l5/l0_l5_X_test.txt")
y_write(y_train, "data/store/l0_l5/l0_l5_y_train.txt")
y_write(y_test, "data/store/l0_l5/l0_l5_y_test.txt")
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

model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['acc'])

history = model.fit(X_train, y_train, batch_size=128, epochs=50, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
predict_list = model.predict(X_test, batch_size=128, verbose=1)
y_predict_write(predict_list, "data/store/l0_l5/l0_l5_predict.txt")
y_write_round(predict_list, "data/store/l0_l5/l0_l5_predict_round.txt")
y_predict_round = []
for index in range(len(predict_list)):
    if float(predict_list[index]) > 0.5:
        y_predict_round.append(1)
    else:
        y_predict_round.append(0)

l0_l5_con_mat = confusion_matrix(list(y_test), y_predict_round)
l0_l5_con_mat_norm = l0_l5_con_mat.astype('float') / l0_l5_con_mat.sum(axis=1)[:, np.newaxis]
l0_l5_con_mat_norm = np.around(l0_l5_con_mat_norm, decimals=6)
# save the result
with open("data/store/l0_l5/l0_l5_result.txt", "a+") as r:
    r.write("\n")
    r.write("\n")
    r.write("epoch is 50 \n")
    r.write("score is " + str(score[0]) + "\n")
    r.write("accuracy is " + str(score[1]) + "\n")
    r.write("\n")
    r.write("node l0_l5 confusion matrix [0, 0]: " + str(l0_l5_con_mat[0, 0]) + "\n")
    r.write("node l0_l5 confusion matrix [0, 1]: " + str(l0_l5_con_mat[0, 1]) + "\n")
    r.write("node l0_l5 confusion matrix [1, 0]: " + str(l0_l5_con_mat[1, 0]) + "\n")
    r.write("node l0_l5 confusion matrix [1, 1]: " + str(l0_l5_con_mat[1, 1]) + "\n")
    r.write("node l0_l5 precision of 0 is " + str(l0_l5_con_mat_norm[0, 0]) + "\n")
    if l0_l5_con_mat[0, 0] == 0.0:
        r.write("node l0_l5 recall of 0 is " + str(0.0) + "\n")
    else:
        r.write("node l0_l5 recall of 0 is " + str(l0_l5_con_mat[0, 0] / (l0_l5_con_mat[0, 0] + l0_l5_con_mat[1, 0])) + "\n")
    r.write("node l0_l5 precision of 1 is " + str(l0_l5_con_mat_norm[1, 1]) + "\n")
    if l0_l5_con_mat[1, 1] == 0.0:
        r.write("node l0_l5 recall of 1 is " + str(0.0) + "\n")
    else:
        r.write("node l0_l5 recall of 1 is " + str(l0_l5_con_mat[1, 1] / (l0_l5_con_mat[0, 1] + l0_l5_con_mat[1, 1])) + "\n")

model.save("models/l0_l5.h5")

epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig("data/img/l0_l5_grad_50.png")
plt.show()

