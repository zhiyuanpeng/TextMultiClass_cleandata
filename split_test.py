import numpy as np
import random


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


def y_write_float(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            s = ""
            for i in range(len(line_value)):
                if i != len(line_value) - 1:
                    s += str(line_value[i])
                    s += " "
                else:
                    s += str(line_value[i])
                    s += "\n"
            f.write(s)


y_test = np.loadtxt("data/processed/y_test.txt", dtype=int)
l0_predict = np.loadtxt("data/store/l0/l0_predict.txt")
l1_predict = np.loadtxt("data/store/l1/l1_predict.txt")
l2_predict = np.loadtxt("data/store/l2/l2_predict.txt")
l3_predict = np.loadtxt("data/store/l3/l3_predict.txt")
l4_predict = np.loadtxt("data/store/l4/l4_predict.txt")
l5_predict = np.loadtxt("data/store/l5/l5_predict.txt")
l0_l1_predict = np.loadtxt("data/store/l0_l1/l0_l1_predict.txt")
l0_l3_predict = np.loadtxt("data/store/l0_l3/l0_l3_predict.txt")
l0_l4_predict = np.loadtxt("data/store/l0_l4/l0_l4_predict.txt")
l0_l5_predict = np.loadtxt("data/store/l0_l5/l0_l5_predict.txt")
l1_l2_predict = np.loadtxt("data/store/l1_l2/l1_l2_predict.txt")
l0_l1_predict_round = np.loadtxt("data/store/l0_l1/l0_l1_predict_round.txt", dtype=int)
l0_l3_predict_round = np.loadtxt("data/store/l0_l3/l0_l3_predict_round.txt", dtype=int)
l0_l4_predict_round = np.loadtxt("data/store/l0_l4/l0_l4_predict_round.txt", dtype=int)
l0_l5_predict_round = np.loadtxt("data/store/l0_l5/l0_l5_predict_round.txt", dtype=int)
l1_l2_predict_round = np.loadtxt("data/store/l1_l2/l1_l2_predict_round.txt", dtype=int)
l0_predict_round = np.loadtxt("data/store/l0/l0_predict_round.txt", dtype=int)
l1_predict_round = np.loadtxt("data/store/l1/l1_predict_round.txt", dtype=int)
l2_predict_round = np.loadtxt("data/store/l2/l2_predict_round.txt", dtype=int)
l3_predict_round = np.loadtxt("data/store/l3/l3_predict_round.txt", dtype=int)
l4_predict_round = np.loadtxt("data/store/l4/l4_predict_round.txt", dtype=int)
l5_predict_round = np.loadtxt("data/store/l5/l5_predict_round.txt", dtype=int)
l0_l1_truth = np.loadtxt("data/split_test/l0_l1_truth.txt", dtype=int)
l0_l3_truth = np.loadtxt("data/split_test/l0_l3_truth.txt", dtype=int)
l0_l4_truth = np.loadtxt("data/split_test/l0_l4_truth.txt", dtype=int)
l0_l5_truth = np.loadtxt("data/split_test/l0_l5_truth.txt", dtype=int)
l1_l2_truth = np.loadtxt("data/split_test/l1_l2_truth.txt", dtype=int)
original_predict = np.loadtxt("data/store/original/original_predict.txt")
original_predict_round = np.loadtxt("data/store/original/original_predict_round.txt")

random.seed(23)
sample_index = random.sample(range(1575), 575)
# get new list
y_test_575 = []
y_test_1000 = []
l0_predict_575 = []
l1_predict_575 = []
l2_predict_575 = []
l3_predict_575 = []
l4_predict_575 = []
l5_predict_575 = []
l0_predict_1000 = []
l1_predict_1000 = []
l2_predict_1000 = []
l3_predict_1000 = []
l4_predict_1000 = []
l5_predict_1000 = []
l0_l1_predict_575 = []
l0_l3_predict_575 = []
l0_l4_predict_575 = []
l0_l5_predict_575 = []
l1_l2_predict_575 = []
l0_l1_predict_round_575 = []
l0_l3_predict_round_575 = []
l0_l4_predict_round_575 = []
l0_l5_predict_round_575 = []
l1_l2_predict_round_575 = []
l0_l1_predict_1000 = []
l0_l3_predict_1000 = []
l0_l4_predict_1000 = []
l0_l5_predict_1000 = []
l1_l2_predict_1000 = []
l0_l1_predict_round_1000 = []
l0_l3_predict_round_1000 = []
l0_l4_predict_round_1000 = []
l0_l5_predict_round_1000 = []
l1_l2_predict_round_1000 = []
l0_predict_round_575 = []
l1_predict_round_575 = []
l2_predict_round_575 = []
l3_predict_round_575 = []
l4_predict_round_575 = []
l5_predict_round_575 = []
l0_predict_round_1000 = []
l1_predict_round_1000 = []
l2_predict_round_1000 = []
l3_predict_round_1000 = []
l4_predict_round_1000 = []
l5_predict_round_1000 = []
l0_l1_truth_575 = []
l0_l3_truth_575 = []
l0_l4_truth_575 = []
l0_l5_truth_575 = []
l1_l2_truth_575 = []
l0_l1_truth_1000 = []
l0_l3_truth_1000 = []
l0_l4_truth_1000 = []
l0_l5_truth_1000 = []
l1_l2_truth_1000 = []
original_predict_575 = []
original_predict_1000 = []
original_predict_round_575 = []
original_predict_round_1000 = []
for index in range(y_test.shape[0]):
    if index in sample_index:
        y_test_575.append((y_test[index]))
        l0_predict_575.append(l0_predict[index])
        l1_predict_575.append(l1_predict[index])
        l2_predict_575.append(l2_predict[index])
        l3_predict_575.append(l3_predict[index])
        l4_predict_575.append(l4_predict[index])
        l5_predict_575.append(l5_predict[index])
        l0_l1_predict_575.append(l0_l1_predict[index])
        l0_l3_predict_575.append(l0_l3_predict[index])
        l0_l4_predict_575.append(l0_l4_predict[index])
        l0_l5_predict_575.append(l0_l5_predict[index])
        l1_l2_predict_575.append(l1_l2_predict[index])
        l0_predict_round_575.append(l0_predict_round[index])
        l1_predict_round_575.append(l1_predict_round[index])
        l2_predict_round_575.append(l2_predict_round[index])
        l3_predict_round_575.append(l3_predict_round[index])
        l4_predict_round_575.append(l4_predict_round[index])
        l5_predict_round_575.append(l5_predict_round[index])
        l0_l1_truth_575.append(l0_l1_truth[index])
        l0_l3_truth_575.append(l0_l3_truth[index])
        l0_l4_truth_575.append(l0_l4_truth[index])
        l0_l5_truth_575.append(l0_l5_truth[index])
        l1_l2_truth_575.append(l1_l2_truth[index])
        original_predict_575.append(original_predict[index])
        original_predict_round_575.append(original_predict_round[index])
        l0_l1_predict_round_575.append(l0_l1_predict_round[index])
        l0_l3_predict_round_575.append(l0_l3_predict_round[index])
        l0_l4_predict_round_575.append(l0_l4_predict_round[index])
        l0_l5_predict_round_575.append(l0_l5_predict_round[index])
        l1_l2_predict_round_575.append(l1_l2_predict_round[index])
    else:
        y_test_1000.append((y_test[index]))
        l0_predict_1000.append(l0_predict[index])
        l1_predict_1000.append(l1_predict[index])
        l2_predict_1000.append(l2_predict[index])
        l3_predict_1000.append(l3_predict[index])
        l4_predict_1000.append(l4_predict[index])
        l5_predict_1000.append(l5_predict[index])
        l0_l1_predict_1000.append(l0_l1_predict[index])
        l0_l3_predict_1000.append(l0_l3_predict[index])
        l0_l4_predict_1000.append(l0_l4_predict[index])
        l0_l5_predict_1000.append(l0_l5_predict[index])
        l1_l2_predict_1000.append(l1_l2_predict[index])
        l0_predict_round_1000.append(l0_predict_round[index])
        l1_predict_round_1000.append(l1_predict_round[index])
        l2_predict_round_1000.append(l2_predict_round[index])
        l3_predict_round_1000.append(l3_predict_round[index])
        l4_predict_round_1000.append(l4_predict_round[index])
        l5_predict_round_1000.append(l5_predict_round[index])
        l0_l1_truth_1000.append(l0_l1_truth[index])
        l0_l3_truth_1000.append(l0_l3_truth[index])
        l0_l4_truth_1000.append(l0_l4_truth[index])
        l0_l5_truth_1000.append(l0_l5_truth[index])
        l1_l2_truth_1000.append(l1_l2_truth[index])
        original_predict_1000.append(original_predict[index])
        original_predict_round_1000.append(original_predict_round[index])
        l0_l1_predict_round_1000.append(l0_l1_predict_round[index])
        l0_l3_predict_round_1000.append(l0_l3_predict_round[index])
        l0_l4_predict_round_1000.append(l0_l4_predict_round[index])
        l0_l5_predict_round_1000.append(l0_l5_predict_round[index])
        l1_l2_predict_round_1000.append(l1_l2_predict_round[index])

x_write(l0_predict_575, "data/split_test/l0_predict_575.txt")
x_write(l1_predict_575, "data/split_test/l1_predict_575.txt")
x_write(l2_predict_575, "data/split_test/l2_predict_575.txt")
x_write(l3_predict_575, "data/split_test/l3_predict_575.txt")
x_write(l4_predict_575, "data/split_test/l4_predict_575.txt")
x_write(l5_predict_575, "data/split_test/l5_predict_575.txt")
x_write(l0_l1_predict_575, "data/split_test/l0_l1_predict_575.txt")
x_write(l0_l3_predict_575, "data/split_test/l0_l3_predict_575.txt")
x_write(l0_l4_predict_575, "data/split_test/l0_l4_predict_575.txt")
x_write(l0_l5_predict_575, "data/split_test/l0_l5_predict_575.txt")
x_write(l1_l2_predict_575, "data/split_test/l1_l2_predict_575.txt")
x_write(l0_l1_predict_round_575, "data/split_test/l0_l1_predict_round_575.txt")
x_write(l0_l3_predict_round_575, "data/split_test/l0_l3_predict_round_575.txt")
x_write(l0_l4_predict_round_575, "data/split_test/l0_l4_predict_round_575.txt")
x_write(l0_l5_predict_round_575, "data/split_test/l0_l5_predict_round_575.txt")
x_write(l1_l2_predict_round_575, "data/split_test/l1_l2_predict_round_575.txt")

x_write(l0_predict_round_575, "data/split_test/l0_predict_round_575.txt")
x_write(l1_predict_round_575, "data/split_test/l1_predict_round_575.txt")
x_write(l2_predict_round_575, "data/split_test/l2_predict_round_575.txt")
x_write(l3_predict_round_575, "data/split_test/l3_predict_round_575.txt")
x_write(l4_predict_round_575, "data/split_test/l4_predict_round_575.txt")
x_write(l5_predict_round_575, "data/split_test/l5_predict_round_575.txt")
x_write(l0_l1_truth_575, "data/split_test/l0_l1_truth_575.txt")
x_write(l0_l3_truth_575, "data/split_test/l0_l3_truth_575.txt")
x_write(l0_l4_truth_575, "data/split_test/l0_l4_truth_575.txt")
x_write(l0_l5_truth_575, "data/split_test/l0_l5_truth_575.txt")
x_write(l1_l2_truth_575, "data/split_test/l1_l2_truth_575.txt")

x_write(l0_predict_1000, "data/split_test/l0_predict_1000.txt")
x_write(l1_predict_1000, "data/split_test/l1_predict_1000.txt")
x_write(l2_predict_1000, "data/split_test/l2_predict_1000.txt")
x_write(l3_predict_1000, "data/split_test/l3_predict_1000.txt")
x_write(l4_predict_1000, "data/split_test/l4_predict_1000.txt")
x_write(l5_predict_1000, "data/split_test/l5_predict_1000.txt")
x_write(l0_l1_predict_1000, "data/split_test/l0_l1_predict_1000.txt")
x_write(l0_l3_predict_1000, "data/split_test/l0_l3_predict_1000.txt")
x_write(l0_l4_predict_1000, "data/split_test/l0_l4_predict_1000.txt")
x_write(l0_l5_predict_1000, "data/split_test/l0_l5_predict_1000.txt")
x_write(l1_l2_predict_1000, "data/split_test/l1_l2_predict_1000.txt")
x_write(l0_l1_predict_round_1000, "data/split_test/l0_l1_predict_round_1000.txt")
x_write(l0_l3_predict_round_1000, "data/split_test/l0_l3_predict_round_1000.txt")
x_write(l0_l4_predict_round_1000, "data/split_test/l0_l4_predict_round_1000.txt")
x_write(l0_l5_predict_round_1000, "data/split_test/l0_l5_predict_round_1000.txt")
x_write(l1_l2_predict_round_1000, "data/split_test/l1_l2_predict_round_1000.txt")
x_write(l0_predict_round_1000, "data/split_test/l0_predict_round_1000.txt")
x_write(l1_predict_round_1000, "data/split_test/l1_predict_round_1000.txt")
x_write(l2_predict_round_1000, "data/split_test/l2_predict_round_1000.txt")
x_write(l3_predict_round_1000, "data/split_test/l3_predict_round_1000.txt")
x_write(l4_predict_round_1000, "data/split_test/l4_predict_round_1000.txt")
x_write(l5_predict_round_1000, "data/split_test/l5_predict_round_1000.txt")
x_write(l0_l1_truth_1000, "data/split_test/l0_l1_truth_1000.txt")
x_write(l0_l3_truth_1000, "data/split_test/l0_l3_truth_1000.txt")
x_write(l0_l4_truth_1000, "data/split_test/l0_l4_truth_1000.txt")
x_write(l0_l5_truth_1000, "data/split_test/l0_l5_truth_1000.txt")
x_write(l1_l2_truth_1000, "data/split_test/l1_l2_truth_1000.txt")


y_write(y_test_575, "data/split_test/y_test_575.txt")
y_write(y_test_1000, "data/split_test/y_test_1000.txt")
y_write_float(original_predict_575, "data/store/original/original_predict_575.txt")
y_write_float(original_predict_1000, "data/store/original/original_predict_1000.txt")
y_write(original_predict_round_575, "data/store/original/original_predict_round_575.txt")
y_write(original_predict_round_1000, "data/store/original/original_predict_round_1000.txt")






