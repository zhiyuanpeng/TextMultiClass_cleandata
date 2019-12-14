import unittest
import numpy as np
import itertools
import viterbi as v
import viterbi_log as vlog


class MyTestCase(unittest.TestCase):
    def test_viterbi_1(self):
        status = [0.9, 0.1, 0.9, 0.3]
        transfer_matrix = np.array([[0, 0.7, 0, 0],
                                    [0.7, 0, 0.4, 0],
                                    [0, 0.4, 0, 0.3],
                                    [0, 0, 0.3, 0]])
        total_score, best_label = v.viterbi(status, transfer_matrix)
        self.assertEqual(total_score, 3.7)
        self.assertEqual(best_label, [1, 1, 1, 0])

    def test_viterbi_2(self):
        status = [0.9, 0.1, 0.9, 0.3, 0.1, 0.9, 0.5]
        transfer_matrix = np.array([[0, 0.7, 0, 0, 0, 0, 0],
                                    [0.7, 0, 0.4, 0, 0, 0, 0],
                                    [0, 0.4, 0, 0.3, 0, 0, 0],
                                    [0, 0, 0.3, 0, 0.2, 0, 0],
                                    [0, 0, 0, 0.2, 0, 0.3, 0],
                                    [0, 0, 0, 0, 0.3, 0, 0.4],
                                    [0, 0, 0, 0, 0, 0.4, 0]])
        total_score, best_label = v.viterbi(status, transfer_matrix)
        self.assertEqual(round(total_score, 2), 6.4)
        self.assertEqual(best_label, [1, 1, 1, 0, 0, 1, 1])

    def test_transfer_label(self):
        trans_value = 0.7
        trans_matrix = v.transfer_label(trans_value)
        right_matrix = np.array([[0, 0], [0, 0.7]])
        for i in range(trans_matrix.shape[0]):
            for j in range(trans_matrix.shape[1]):
                self.assertEqual(trans_matrix[i, j], right_matrix[i, j])

    def test_re_order_transfer(self):
        transfer_matrix = np.array([[0, 0.7, 0, 0, 0, 0, 0],
                                    [0.7, 0, 0.4, 0, 0, 0, 0],
                                    [0, 0.4, 0, 0.3, 0, 0, 0],
                                    [0, 0, 0.3, 0, 0.2, 0, 0],
                                    [0, 0, 0, 0.2, 0, 0.3, 0],
                                    [0, 0, 0, 0, 0.3, 0, 0.4],
                                    [0, 0, 0, 0, 0, 0.4, 0]])
        new_index = [6, 1, 0, 3, 5, 4, 2]
        new_matrix = v.re_order_transfer(transfer_matrix, new_index)
        right_matrix = np.array([[0, 0, 0, 0, 0.4, 0, 0],
                                 [0, 0, 0.7, 0, 0, 0, 0.4],
                                 [0, 0.7, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0.2, 0.3],
                                 [0.4, 0, 0, 0, 0, 0.3, 0],
                                 [0, 0, 0, 0.2, 0.3, 0, 0],
                                 [0, 0.4, 0, 0.3, 0, 0, 0]])
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                self.assertEqual(new_matrix[i, j], right_matrix[i, j])

    def test_viterbi_l1(self):
        # l0, l1, l2
        status = [0.1, 0.7, 0.6]
        transfer_matrix = np.array([[0, 0.7, 0],
                                    [0.7, 0, 0.2],
                                    [0, 0.2, 0]])
        new_index = [0, 1, 2]
        new_matrix = v.re_order_transfer(transfer_matrix, new_index)
        right_matrix = np.array([[0, 0.7, 0],
                                 [0.7, 0, 0.2],
                                 [0, 0.2, 0]])
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                self.assertEqual(new_matrix[i, j], right_matrix[i, j])
        total_score, best_label = v.viterbi(status, new_matrix)
        self.assertEqual(round(total_score, 2), 2.4)
        self.assertEqual(best_label, [0, 1, 1])

    def test_viterbi_l2(self):
        # l0, l2, l1
        status = [0.1, 0.6, 0.7]
        transfer_matrix = np.array([[0, 0.7, 0],
                                    [0.7, 0, 0.2],
                                    [0, 0.2, 0]])
        new_index = [0, 2, 1]
        new_matrix = v.re_order_transfer(transfer_matrix, new_index)
        right_matrix = np.array([[0, 0, 0.7],
                                 [0, 0, 0.2],
                                 [0.7, 0.2, 0]])
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                self.assertEqual(new_matrix[i, j], right_matrix[i, j])
        total_score, best_label = v.viterbi(status, new_matrix)
        self.assertEqual(round(total_score, 2), 2.4)
        self.assertEqual(best_label, [0, 1, 1])

    def test_viterbi_l3(self):
        # l1, l0, l2
        status = [0.7, 0.1, 0.6]
        transfer_matrix = np.array([[0, 0.7, 0],
                                    [0.7, 0, 0.2],
                                    [0, 0.2, 0]])
        new_index = [1, 0, 2]
        new_matrix = v.re_order_transfer(transfer_matrix, new_index)
        right_matrix = np.array([[0, 0.7, 0.2],
                                 [0.7, 0, 0],
                                 [0.2, 0, 0]])
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                self.assertEqual(new_matrix[i, j], right_matrix[i, j])
        total_score, best_label = v.viterbi(status, new_matrix)
        self.assertEqual(round(total_score, 2), 2.2)
        self.assertEqual(best_label, [1, 0, 1])

    def test_viterbi_l4(self):
        # l1, l2, l0
        status = [0.7, 0.6, 0.1]
        transfer_matrix = np.array([[0, 0.7, 0],
                                    [0.7, 0, 0.2],
                                    [0, 0.2, 0]])
        new_index = [1, 2, 0]
        new_matrix = v.re_order_transfer(transfer_matrix, new_index)
        right_matrix = np.array([[0, 0.2, 0.7],
                                 [0.2, 0, 0],
                                 [0.7, 0, 0]])
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                self.assertEqual(new_matrix[i, j], right_matrix[i, j])
        total_score, best_label = v.viterbi(status, new_matrix)
        self.assertEqual(round(total_score, 2), 2.4)
        self.assertEqual(best_label, [1, 1, 0])

    def test_viterbi_l5(self):
        # l2, l0, l1
        status = [0.6, 0.1, 0.7]
        transfer_matrix = np.array([[0, 0.7, 0],
                                    [0.7, 0, 0.2],
                                    [0, 0.2, 0]])
        new_index = [2, 0, 1]
        new_matrix = v.re_order_transfer(transfer_matrix, new_index)
        right_matrix = np.array([[0, 0, 0.2],
                                 [0, 0, 0.7],
                                 [0.2, 0.7, 0]])
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                self.assertEqual(new_matrix[i, j], right_matrix[i, j])
        total_score, best_label = v.viterbi(status, new_matrix)
        self.assertEqual(round(total_score, 2), 2.2)
        self.assertEqual(best_label, [1, 0, 1])

    def test_viterbi_l6(self):
        # l2, l1, l0
        status = [0.6, 0.7, 0.1]
        transfer_matrix = np.array([[0, 0.7, 0],
                                    [0.7, 0, 0.2],
                                    [0, 0.2, 0]])
        new_index = [2, 1, 0]
        new_matrix = v.re_order_transfer(transfer_matrix, new_index)
        right_matrix = np.array([[0, 0.2, 0],
                                 [0.2, 0, 0.7],
                                 [0, 0.7, 0]])
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                self.assertEqual(new_matrix[i, j], right_matrix[i, j])
        total_score, best_label = v.viterbi(status, new_matrix)
        self.assertEqual(round(total_score, 2), 2.4)
        self.assertEqual(best_label, [1, 1, 0])

    def test_permutation_optimal(self):
        status = [0.1, 0.7, 0.6]
        transfer_matrix = np.array([[0, 0.7, 0],
                                    [0.7, 0, 0.2],
                                    [0, 0.2, 0]])
        best_label = v.permutation_optimal(status, transfer_matrix)
        self.assertEqual(best_label, [0, 1, 1])

    def test_vlog_permutation_optimal(self):
        status = [np.log(0.98), np.log(0.3), np.log(0.88), np.log(0.02), np.log(0.4), np.log(0.008)]
        transfer_matrix = np.array([[-1000, np.log(0.95), -1000, np.log(0.92), np.log(0.92), np.log(0.97)],
                                   [np.log(0.95), -1000, np.log(0.92), -1000, -1000, -1000],
                                   [-1000, np.log(0.92), -1000, -1000, -1000, -1000],
                                   [np.log(0.92), -1000, -1000, -1000, -1000, -1000],
                                   [np.log(0.92), -1000, -1000, -1000, -1000, -1000],
                                   [np.log(0.92), -1000, -1000, -1000, -1000, -1000]])
        best_label = vlog.permutation_optimal(status, transfer_matrix)
        print(best_label)
        print("done")


if __name__ == '__main__':
    unittest.main()
