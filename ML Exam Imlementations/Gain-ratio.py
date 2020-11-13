import numpy as np
from math import log2
from library import information_gain


# Gain ratio
#
# You are given a np.array of n elements with either labels 0 (negative) of 1 (positive) labels.
# And a np.array of a discrete attribute. Assuming that in a decision tree,
# the labels would be split according to the values of this attribute.
# Implement the split_entropy() and gain_ratio() function to calculate the gain ratio.
#
# The formula for split entropy is:
#
# SE= − ∑V∈Values(A) |SV| / |S| log2 |SV| / |S|

# Hint: information gain is already implemented in the library
# Hint: to call a function from the Solution class you can use: Solution.function()


class Solution():

    def split_entropy(attribute, labels):
        n = len(attribute)
        unique_attributes = np.unique(attribute)

        SE = 0

        for a in unique_attributes:
            n_a = len(np.where(attribute == a)[0])
            SE -= (n_a/n)* log2(n_a / n)

        return SE

    def gain_ratio(attribute, labels):

        IG = information_gain(attribute, labels)
        SE = Solution.split_entropy(attribute, labels)
        gain_ratio =  IG/SE

        return gain_ratio
