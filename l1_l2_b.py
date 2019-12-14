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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# read the train data
toxic_comments = pd.read_csv("data/unprocess/train.csv")
filter = toxic_comments["comment_text"] != ""
toxic_comments = toxic_comments[filter]
toxic_comments = toxic_comments.dropna()
# we only train a binary classifier for lable 1
# [l0, l1, l2, l3, l4, l5]
# ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = np.array(toxic_comments[["toxic"]])*np.array(toxic_comments[["severe_toxic"]])


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
non_zero = np.sum(y_train)
rate = non_zero/(len(y_train) - non_zero)
X_train_new = []
y_train_new = []
np.random.seed(0)
for i in range(y_train.shape[0]):
    if y_train[i, 0] == 1:
        X_train_new.append(X_train[i])
        y_train_new.append([1])
    elif np.random.random() < rate:
        X_train_new.append(X_train[i])
        y_train_new.append([0])
y_train_new = np.array(y_train_new)
# write the X_train, X_test, y_train, y_test
x_write(X_train_new, "data/store/l1_l2_b_X_train.txt")
x_write(X_test, "data/store/l1_l2_b_X_test.txt")
y_write(y_train_new, "data/store/l1_l2_b_y_train.txt")
y_write(y_test, "data/store/l1_l2_b_y_test.txt")
#
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train_new)

X_train_new = tokenizer.texts_to_sequences(X_train_new)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train_new = pad_sequences(X_train_new, padding='post', maxlen=maxlen)
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

history = model.fit(X_train_new, y_train_new, batch_size=128, epochs=500, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
predict_list = model.predict(X_test, batch_size=128, verbose=1)
y_write(predict_list, "data/store/l1_l2_b_predict.txt")
y_write_round(predict_list, "data/store/l1_l2_b_predict_round.txt")
right_sum = 0
for index in range(len(predict_list)):
    if float(predict_list[index]) > 0.5:
        y_predict = 1
    else:
        y_predict = 0
    if y_predict == int(y_test[index]):
        right_sum += 1
accuracy = (right_sum/len(predict_list))*100

round_predict_list = []
for index in range(len(predict_list)):
    if float(predict_list[index]) > 0.5:
        round_predict_list.append(1)
    else:
        round_predict_list.append(0)
con_mat = confusion_matrix(list(y_test), round_predict_list)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=6)
# save the result
with open("data/store/l1_l2_b_result.txt", "a+") as r:
    r.write("self accuracy is " + str(accuracy) + "\n")
    r.write("score is " + str(score[0]) + "\n")
    r.write("accuracy is " + str(score[1]) + "\n")
    r.write("total number of training data is " + str(len(y_train_new)) + "\n")
    r.write("zero label in training dataset " + str(np.sum(y_train_new)) + "\n")
    r.write("precision of 0 is " + str(con_mat_norm[0, 0]) + "\n")
    r.write("precision of 1 is " + str(con_mat_norm[1, 1]) + "\n")

model.save("models/l1_l2_b.h5")

epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig("data/img/l1_l2_b_grad_500.png")
plt.show()

