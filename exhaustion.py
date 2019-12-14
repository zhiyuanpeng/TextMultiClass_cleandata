import numpy as np
import itertools
import viterbi as v


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


def overall_optimal(status_list, transfer_matrix):
    score_list = []
    all_possible = get_candidate(len(status_list))
    for one_possible in all_possible:
        score = 0
        for index in range(len(one_possible)):
            if index != 0 and one_possible[index - 1] == 1 and one_possible[index] == 1:
                score += transfer_matrix[index - 1, index]
            score += (status_list[index] if one_possible[index] == 1 else (1 - status_list[index]))
        # score = score/np.sum(one_possible) if np.sum(one_possible) != 0 else score
        score_list.append(score)
    max_score = max(score_list)
    max_index = score_list.index(max_score)
    max_label = all_possible[max_index]
    return max_score, max_label


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
        per_transfer = v.re_order_transfer(transfer_matrix, index_list[index])
        score, label = overall_optimal(permutation_list[index], per_transfer)
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
    node_path_list = ["l0_predict.txt", "l1_predict.txt", "l2_predict.txt", "l3_predict.txt", "l4_predict.txt"
                      , "l5_predict.txt"]
    link_path_list = ["l0_l1_predict.txt", "l0_l3_predict.txt", "l0_l4_predict.txt", "l0_l5_predict.txt"
                      , "l1_l2_predict.txt"]
    node_matrix = v.get_node_matrix(node_path_list)
    transfer_list = v.get_transfer_matrix(link_path_list)
    with open("data/store/exhaustion_result_length.txt", "a+") as vf:
        iteration = 0
        for index in range(node_matrix.shape[0]):
            best_label = permutation_optimal(node_matrix[index, :], transfer_list[index])
            s = ""
            for i in range(len(best_label) - 1):
                s = s + str(int(best_label[i])) + " "
            s += str(int(best_label[len(best_label) - 1]))
            if iteration < 2000:
                vf.write(s + "\n")
            else:
                break
            iteration += 1
        print(index)


if __name__ == '__main__':
    main()
