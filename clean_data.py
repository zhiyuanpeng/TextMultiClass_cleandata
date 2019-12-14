from sklearn.model_selection import train_test_split

import pandas as pd
import re
import numpy as np

import os
# read the train data
toxic_comments = pd.read_csv("data/unprocess/train.csv")
filter = toxic_comments["comment_text"] != ""
toxic_comments = toxic_comments[filter]
toxic_comments = toxic_comments.dropna()
# we only train a binary classifier for lable 1
# [l0, l1, l2, l3, l4, l5]
# ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
toxic_comments_labels = toxic_comments[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]


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
            s = ""
            for i in range(len(line_value)):
                if i != len(line_value) - 1:
                    s += str(int(line_value[i]))
                    s += " "
                else:
                    s += str(int(line_value[i]))
                    s += "\n"
            f.write(s)


X = []
sentences = list(toxic_comments["comment_text"])
for sen in sentences:
    X.append(preprocess_text(sen))

y = toxic_comments_labels.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
# remove the all zero data
X_train_clean = []
y_train_clean = np.zeros((0, y_train.shape[1]))
X_test_clean = []
y_test_clean = np.zeros((0, y_test.shape[1]))
for i in range(y_train.shape[0]):
    if (y_train[i, :] == 0).all():
        continue
    else:
        X_train_clean.append(X_train[i])
        y_train_clean = np.vstack((y_train_clean, y_train[i, :]))

for i in range(y_test.shape[0]):
    if (y_test[i, :] == 0).all():
        continue
    else:
        X_test_clean.append(X_test[i])
        y_test_clean = np.vstack((y_test_clean, y_test[i, :]))

# write the X_train, X_test, y_train, y_test
x_write(X_train_clean, "data/processed/X_train.txt")
x_write(X_test_clean, "data/processed/X_test.txt")
y_write(y_train_clean, "data/processed/y_train.txt")
y_write(y_test_clean, "data/processed/y_test.txt")
