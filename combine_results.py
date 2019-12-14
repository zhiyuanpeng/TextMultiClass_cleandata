import numpy as np
#  combine the result of each binary classifier


def get_node_matrix(node_list):
    # get the matrix size
    path = "./data/store/"
    example = list(np.loadtxt(path + node_list[0]))
    node_matrix = np.zeros((len(example), len(node_list)))
    for index in range(len(node_list)):
        node_column = list(np.loadtxt(path + node_list[index]))
        node_matrix[:, index] = node_column
    return node_matrix


def get_transfer_matrix(link_list):
    path = "./data/store/"
    transfer_list = []
    # convert the file to matrix
    transfer_data = get_node_matrix(link_list)
    for line in transfer_data:
        transfer_matrix = np.zeros((len(line) + 1, len(line) + 1))
        transfer_matrix[0, 1] = line[0]
        transfer_matrix[0, 3] = line[1]
        transfer_matrix[0, 4] = line[2]
        transfer_matrix[0, 5] = line[3]
        transfer_matrix[1, 2] = line[4]
        #
        transfer_matrix[1, 0] = line[0]
        transfer_matrix[3, 0] = line[1]
        transfer_matrix[4, 0] = line[2]
        transfer_matrix[5, 0] = line[3]
        transfer_matrix[2, 1] = line[4]
        transfer_list.append(transfer_matrix.copy())
    return transfer_list


def main():

    node_path_list = ["l0_predict.txt", "l1_predict.txt", "l2_predict.txt", "l3_predict.txt", "l4_predict.txt"
                      , "l5_predict.txt"]
    link_path_list = ["l0_l1_predict.txt", "l0_l3_predict.txt", "l0_l4_predict.txt", "l0_l5_predict.txt"
                      , "l1_l2_predict.txt"]
    node_matrix = get_node_matrix(node_path_list)
    transfer_list = get_transfer_matrix(link_path_list)
    print("done")


if __name__ == '__main__':
    main()

