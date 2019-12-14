import pandas as pd
import numpy as np

toxic_comments = pd.read_csv("data/unprocess/train.csv")
filter = toxic_comments["comment_text"] != ""
toxic_comments = toxic_comments[filter]
toxic_comments = toxic_comments.dropna()
# we only train a binary classifier for lable 1
# [l0, l1, l2, l3, l4, l5]
# ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
l0_l1 = np.array(toxic_comments[["toxic"]])*np.array(toxic_comments[["severe_toxic"]])
l1_l2 = np.array(toxic_comments[["severe_toxic"]])*np.array(toxic_comments[["obscene"]])
l0_l3 = np.array(toxic_comments[["toxic"]])*np.array(toxic_comments[["threat"]])
l0_l4 = np.array(toxic_comments[["toxic"]])*np.array(toxic_comments[["insult"]])
l0_l5 = np.array(toxic_comments[["toxic"]])*np.array(toxic_comments[["identity_hate"]])

print(np.sum(l0_l1)/len(l0_l1))
print(np.sum(l1_l2)/len(l0_l1))
print(np.sum(l0_l3)/len(l0_l1))
print(np.sum(l0_l4)/len(l0_l1))
print(np.sum(l0_l5)/len(l0_l1))
print(len(l0_l1))
