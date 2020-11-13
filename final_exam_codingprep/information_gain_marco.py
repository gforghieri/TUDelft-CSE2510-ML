import numpy as np
from math import log2


class Solution():

    def information_gain(labels, subsets):
        a = Solution.entropy(labels)
        b = Solution.split_entropy(labels, subsets)
        return a - b

    def entropy(labels):
        length = len(labels)
        totalPositive = labels.sum()
        totalNegative = length - totalPositive
        positiveSubset = totalPositive / length
        negativeSubset = totalNegative / length
        positiveEntropy = 0
        if (positiveSubset > 0):
            positiveEntropy = - positiveSubset * log2(positiveSubset)
        negativeEntropy = 0
        if (negativeSubset > 0):
            negativeEntropy = - negativeSubset * log2(negativeSubset)
        return positiveEntropy + negativeEntropy

    def split_entropy(labels, subsets):
        length = len(labels)
        uniques = np.unique(subsets)
        data = 0
        for i in uniques:
            labelsOfSubsets = labels[subsets == i]
            data += len(labelsOfSubsets) / length * Solution.entropy(labelsOfSubsets)
        return data
