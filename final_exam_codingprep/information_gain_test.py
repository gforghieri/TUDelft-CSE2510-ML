import numpy as np

import unittest
# from solution import Solution
# import weblabTestRunner
from information_gain import Solution


class TestSolution(unittest.TestCase):
    # Place all the tests between the START comment and the END comment.
    # Do not remove the SPECTESTS comments

    # SPECTESTS START HERE

    def test_information_gain_case_0(self):
        labels = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1])
        grouping = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.0, places=2)

    def test_information_gain_case_1(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
        grouping = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.0, places=2)

    def test_information_gain_case_2(self):
        labels = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 0])
        grouping = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.0, places=2)

    def test_information_gain_case_3(self):
        labels = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
        grouping = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.0, places=2)

    def test_information_gain_case_4(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1])
        grouping = np.array([1, 3, 1, 2, 2, 3, 1, 1, 4, 4, 2, 4, 2, 4, 3, 3, 3, 4, 0, 0, 3, 0, 4, 2, 4, 4, 4, 2, 0, 2])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.019256415477946653, places=2)

    def test_information_gain_case_5(self):
        labels = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        grouping = np.array([3, 4, 1, 4, 0, 1, 0, 2, 3, 4, 2, 0, 3, 3, 2, 0, 0, 1, 0, 2, 3, 3, 3, 1, 4, 3, 1, 4, 1, 4])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.15315277408372385, places=2)

    def test_information_gain_case_6(self):
        labels = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1])
        grouping = np.array([1, 3, 2, 2, 1, 1, 4, 0, 1, 2, 4, 4, 3, 3, 2, 4, 3, 1, 1, 1, 4, 1, 0, 4, 1, 4, 2, 4, 0, 3])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.07572997419592853, places=2)

    def test_information_gain_case_7(self):
        labels = np.array(
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0,
             1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        grouping = np.array(
            [122, 14, 122, 31, 14, 122, 14, 128, 43, 178, 122, 31, 43, 43, 14, 14, 174, 122, 134, 134, 129, 134, 14,
             122, 128, 31, 134, 134, 122, 122, 14, 128, 128, 129, 43, 134, 43, 128, 129, 134, 31, 122, 31, 178, 14, 178,
             122, 122, 174, 129])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.174713446403994, places=2)

    def test_information_gain_case_8(self):
        labels = np.array(
            [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
             1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0])
        grouping = np.array(
            [78, 169, 178, 121, 121, 22, 47, 22, 47, 32, 169, 123, 178, 123, 178, 47, 121, 32, 47, 169, 47, 32, 178,
             121, 121, 121, 123, 169, 78, 169, 32, 123, 78, 23, 178, 32, 47, 47, 47, 78, 123, 121, 169, 47, 47, 32, 169,
             178, 47, 47])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.09961496562499339, places=2)

    def test_information_gain_case_9(self):
        labels = np.array(
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
             0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0])
        grouping = np.array(
            [130, 98, 130, 36, 64, 115, 66, 64, 98, 178, 130, 130, 66, 90, 178, 64, 115, 66, 115, 66, 115, 178, 66, 98,
             178, 130, 130, 98, 90, 36, 178, 115, 66, 66, 36, 66, 98, 66, 178, 115, 130, 66, 130, 66, 115, 115, 130,
             115, 178, 98])
        self.assertAlmostEqual(Solution.information_gain(labels, grouping), 0.05891247318343129, places=2)


# SPECTESTS END HERE

#
# if __name__ == "__main__":
#     unittest.main(testRunner=weblabTestRunner.TestRunner)

