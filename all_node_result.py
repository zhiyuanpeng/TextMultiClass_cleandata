import numpy as np
#  combine the result of each binary classifier


def get_node_matrix(node_list):
    # get the matrix size
    path = "./data/split_test/"
    example = list(np.loadtxt(path + node_list[0], dtype=int))
    node_matrix = np.zeros((len(example), len(node_list)), dtype=int)
    for index in range(len(node_list)):
        node_column = list(np.loadtxt(path + node_list[index], dtype=int))
        node_matrix[:, index] = node_column
    return node_matrix


def y_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for row in range(list_name.shape[0]):
            s = ""
            for column in range(list_name.shape[1]):
                s += str(list_name[row, column])
                if column != (list_name.shape[1] - 1):
                    s += " "
            f.write(s + "\n")


def main():

    node_path_list = ["l0_predict_round_1000.txt", "l1_predict_round_1000.txt", "l2_predict_round_1000.txt",
                      "l3_predict_round_1000.txt", "l4_predict_round_1000.txt", "l5_predict_round_1000.txt"]
    node_matrix = get_node_matrix(node_path_list)
    y_write(node_matrix, "data/elbow/all_node_round_1000.txt")


if __name__ == '__main__':
    main()

