import numpy as np

y_test = np.loadtxt("data/processed/y_test.txt", dtype=int)
l0_l1 = [y_test[i, 0]*y_test[i, 1] for i in range(y_test.shape[0])]
l0_l3 = [y_test[i, 0]*y_test[i, 3] for i in range(y_test.shape[0])]
l0_l4 = [y_test[i, 0]*y_test[i, 4] for i in range(y_test.shape[0])]
l0_l5 = [y_test[i, 0]*y_test[i, 5] for i in range(y_test.shape[0])]
l1_l2 = [y_test[i, 1]*y_test[i, 2] for i in range(y_test.shape[0])]

with open("data/split_test/l0_l1_truth.txt", "a+") as edge0_1:
    for value in l0_l1:
        edge0_1.write(str(value) + "\n")

with open("data/split_test/l0_l3_truth.txt", "a+") as edge0_3:
    for value in l0_l3:
        edge0_3.write(str(value) + "\n")

with open("data/split_test/l0_l4_truth.txt", "a+") as edge0_4:
    for value in l0_l4:
        edge0_4.write(str(value) + "\n")

with open("data/split_test/l0_l5_truth.txt", "a+") as edge0_5:
    for value in l0_l5:
        edge0_5.write(str(value) + "\n")

with open("data/split_test/l1_l2_truth.txt", "a+") as edge1_2:
    for value in l1_l2:
        edge1_2.write(str(value) + "\n")
