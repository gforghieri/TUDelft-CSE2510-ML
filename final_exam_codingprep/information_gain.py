import numpy as np
from math import log2


class Solution():

    def information_gain(labels, subsets):
        ent = Solution.entropy(labels)
        Im = Solution.split_entropy(labels, subsets)
        return ent - Im

    def entropy(labels):
        n = len(labels)

        # calculate the fraction 1s and 0s
        total_1s = labels.sum()
        frac_1s = total_1s / n
        total_0s = n - total_1s
        frac_0s = total_0s / n

        # get individual entropies
        # entropy_1s
        entropy_1s = 0
        if frac_1s > 0:
            entropy_1s = - frac_1s * log2(frac_1s)
        # entropy_0s
        entropy_0s = 0
        if frac_0s > 0:
            entropy_0s = - frac_0s * log2(frac_0s)

        # get total entropy
        return entropy_1s + entropy_0s

    def split_entropy(labels, subsets):

        n = len(labels)
        information = 0
        unique_subsets = np.unique(subsets)

        for i in unique_subsets:
            subset_labels = labels[subsets == i]
            information += len(subset_labels) / n * Solution.entropy(subset_labels)

        return information