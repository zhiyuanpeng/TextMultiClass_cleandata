import numpy as np
import itertools


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


def list_expand(status_list, direction):
    expand_matrix = np.zeros((len(status_list), len(status_list)))
    for i in range(len(status_list)):
        for j in range(len(status_list)):
            if direction == 0:
                # vertical expand
                # list[0] list[1]
                # list[0] list[1]
                expand_matrix[i, j] = status_list[j]
                expand_matrix[i, j] = status_list[j]
            else:
                # horizontal
                # list[0] list[0]
                # list[1] list[1]
                expand_matrix[i, j] = status_list[i]
                expand_matrix[i, j] = status_list[i]
    return expand_matrix


def transfer_label(trans_value):
    """
    trans_value is the possibility of 1 1, we need to get
       0 1
     0 x x
     1 x x
    """
    label_trans_matrix = np.zeros((2, 2))
    label_trans_matrix[0, 0] = 0
    label_trans_matrix[0, 1] = 0
    label_trans_matrix[1, 0] = 0
    label_trans_matrix[1, 1] = trans_value
    return label_trans_matrix


def viterbi(status, transfer_matrix):
    previous = []
    obs = []
    best_label = []
    # store_index will begin from the second statue
    # [1, 1] means the previous label is 1 for both the two current status 0 and 1
    store_index = []
    store_score = []
    for index in range(len(status)):
        # get the current status
        obs = [1 - status[index], status[index]]
        # for the first state
        # the index is 0
        if len(previous) == 0:
            # no transfer_matrix info will be used
            previous = [1 - status[index], status[index]]
        # from the second the status, we need use the transfer matrix info
        else:
            # get the trans_label
            #   0 1
            # 0 x x
            # 1 x x
            label_trans_matrix = transfer_label(transfer_matrix[index - 1, index])
            previous_expand = list_expand(previous, 1)
            obs_expand = list_expand(obs, 0)
            scores = previous_expand + obs_expand + label_trans_matrix
            previous = [max(scores[0, 0], scores[1, 0]), max(scores[0, 1], scores[1, 1])]
            # store the info of the path
            store_index.append((list(scores[:, 0]).index(previous[0]), list(scores[:, 1]).index(previous[1])))
    # after the forward calculation, we need go back to find the optimal path
    total_score = max(previous[0], previous[1])
    # get the label for the last state
    best_label.append(previous.index(total_score))
    for i in range(len(store_index) - 1, -1, -1):
        # from the label of the last state we go back to find the path
        best_label.append(store_index[i][best_label[len(store_index) - i - 1]])
    return total_score, [best_label[i] for i in range(len(best_label) - 1, -1, -1)]


def re_order_transfer(transfer_matrix, new_index):
    """
    for each permutation of the status, need re order the transfer matrix to math the status
    @param transfer_matrix:
    @param new_index:
    @return: re-ordered transfer_matrix
    """
    new_matrix = np.zeros_like(transfer_matrix)
    for row in range(new_matrix.shape[0]):
        for column in range(new_matrix.shape[1]):
            old_row = new_index[row]
            old_column = new_index[column]
            new_matrix[row, column] = transfer_matrix[old_row, old_column]
    return new_matrix


def permutation_optimal(status, transfer_matrix):
    """
    calculate the score of each permutation of the input status
    output the label with the highest score
    """
    # get all the permutations
    permutation_list = list(itertools.permutations(status, len(status)))
    index_list = list(itertools.permutations([i for i in range(len(status))], len(status)))
    score_list = []
    label_list = []
    for index in range(len(permutation_list)):
        # the order of status changed, so as the order of the transfer_matrix
        per_transfer = re_order_transfer(transfer_matrix, index_list[index])
        score, label = viterbi(permutation_list[index], per_transfer)
        score_list.append(score)
        label_list.append(label)
    # select the label with the highest score
    max_score = max(score_list)
    best_label = [0 for i in range(len(status))]
    for index in range(len(score_list)):
        if score_list[index] == max_score:
            # recover the order and get the label from
            # label_list[index] index_list[index]
            for i in range(len(index_list[index])):
                best_label[index_list[index][i]] = label_list[index][i]
            # once we find the max, we don't need to care about the second same max
            break
    return best_label


def main():
    link_path_list = ["l0_l1_predict.txt", "l0_l3_predict.txt", "l0_l4_predict.txt", "l0_l5_predict.txt"
                      , "l1_l2_predict.txt"]
    node_matrix = np.loadtxt("data/store/original_predict.txt")
    transfer_list = get_transfer_matrix(link_path_list)
    print(node_matrix.shape)

    with open("data/store/viterbi_plus_original_result.txt", "a+") as vf:
        for index in range(node_matrix.shape[0]):
            best_label = permutation_optimal(node_matrix[index, :], transfer_list[index])
            s = ""
            for i in range(len(best_label) - 1):
                s = s + str(int(best_label[i])) + " "
            s += str(int(best_label[len(best_label) - 1]))
            vf.write(s + "\n")
        print(index)


if __name__ == '__main__':
    main()