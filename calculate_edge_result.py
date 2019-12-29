import numpy as np


viterbi_result = np.array(np.loadtxt("./data/store/original/original_predict_575.txt", dtype=int))
label = np.loadtxt("./data/split_test/y_test_575.txt", dtype=int)
total_right = 0
for i in range(viterbi_result.shape[0]):
    if viterbi_result[i, ] == label[i, ]:
        total_right += 1

print("edge accuracy is: " + str(total_right/label.shape[0]) + "\n")
