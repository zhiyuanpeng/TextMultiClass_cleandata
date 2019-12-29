import numpy as np
import itertools
import viterbi as v
import math
import argparse


def bit_to_list(t, n):
    """
    convert an int to list
    @param t: the int
    @param n: the length of the bit
    @return: a lit
    """
    bit_list = [0 for i in range(n)]
    i = -1
    while t != 0:
        bit_list[i] = t % 2
        t = t >> 1
        i -= 1
    return bit_list


def get_candidate(list_length):
    """
    the length of the list
    @param list_length:
    @return: list contains all the possible
    """
    candidate = []
    for i in range(pow(2, list_length)):
        candidate.append(bit_to_list(i, list_length))
    return candidate


def transfer_label(trans_value):
    """
    trans_value is the possibility of 1 1, we need to get
       0 1
     0 x x
     1 x x
    """
    label_trans_matrix = np.zeros((2, 2))
    label_trans_matrix[0, 0] = -1000
    label_trans_matrix[0, 1] = -1000
    label_trans_matrix[1, 0] = -1000
    label_trans_matrix[1, 1] = trans_value
    return label_trans_matrix


def overall_optimal(status_list, transfer_matrix, lambda_value):
    score_list = []
    all_possible = get_candidate(len(status_list))
    for one_possible in all_possible:
        score = 0
        for index in range(len(one_possible)):
            if index != 0:
                label_trans_matrix = transfer_label(transfer_matrix[index - 1, index])
                score += label_trans_matrix[one_possible[index - 1], one_possible[index]]
            score += (status_list[index] if one_possible[index] == 1 else np.log(1 - math.exp(status_list[index])))
        penalty = lambda_value*np.sum(one_possible)
        score += lambda_value*np.sum(one_possible)
        score_list.append(score)
    max_score = max(score_list)
    max_index = score_list.index(max_score)
    max_label = all_possible[max_index]
    return max_score, max_label


def permutation_optimal(status, transfer_matrix, lambda_value):
    """
    calculate the score of each permutation of the input status
    output the label with the highest score
    """
    # get all the permutations
    permutation_list = list(itertools.permutations(status, len(status)))
    index_list = list(itertools.permutations([i for i in range(len(status))], len(status)))
    score_list = []
    label_list = []
    deg = 0
    for index in range(len(permutation_list)):
        deg += 1
        # the order of status changed, so as the order of the transfer_matrix
        per_transfer = v.re_order_transfer(transfer_matrix, index_list[index])
        score, label = overall_optimal(permutation_list[index], per_transfer, lambda_value)
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


def exhust_search(filename, lambda_value):
    node_path_list = ["l0_predict_575.txt", "l1_predict_575.txt", "l2_predict_575.txt", "l3_predict_575.txt",
                      "l4_predict_575.txt", "l5_predict_575.txt"]

    # link_path_list = ["l0_l1_predict_575.txt", "l0_l3_predict_575.txt", "l0_l4_predict_575.txt",
    #                   "l0_l5_predict_575.txt", "l1_l2_predict_575.txt"]
    link_path_list = ["l0_l1_truth_575.txt", "l0_l3_truth_575.txt", "l0_l4_truth_575.txt",
                      "l0_l5_truth_575.txt", "l1_l2_truth_575.txt"]
    # node_matrix = v.get_node_matrix(node_path_list)
    node_matrix = np.loadtxt("data/store/original/original_predict_575.txt")
    transfer_list = v.get_transfer_matrix(link_path_list)
    with open("data/exhaustion/" + filename, "a+") as vf:
        iteration = 0
        for index in range(node_matrix.shape[0]):
            node_matrix_log = np.zeros_like(node_matrix[index, :])
            for i in range(len(node_matrix[index, :])):
                node_matrix_log[i, ] = np.log(node_matrix[index, i])
            transfer_list_log = np.zeros_like(transfer_list[index])
            for i in range(transfer_list[index].shape[0]):
                for j in range(transfer_list[index].shape[1]):
                    if transfer_list[index][i, j] != 0:
                        transfer_list_log[i, j] = np.log(transfer_list[index][i, j])
                    else:
                        transfer_list_log[i, j] = -1000
            best_label = permutation_optimal(node_matrix_log, transfer_list_log, lambda_value)
            s = ""
            for i in range(len(best_label) - 1):
                s = s + str(int(best_label[i])) + " "
            s += str(int(best_label[len(best_label) - 1]))
            # if iteration < 2000:
            vf.write(s + "\n")
            # else:
            #     break
            # iteration += 1
        print(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="file name of the result file")
    parser.add_argument("lambda_value", type=float, help="lambda * #of 1s")
    args = parser.parse_args()
    exhust_search(args.file_name, args.lambda_value)


if __name__ == '__main__':
    main()
