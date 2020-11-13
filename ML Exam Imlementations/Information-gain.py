# import numpy as np
# from math import log2
#
# # You are given a list of n elements consisting of labels 0 (negative) and 1 (positive) labels.
# # These are split up by a classifier in 1 or multiple subsets. The set every element belongs to is indicated in sets.
# #
# # E.g. if we have the following labels: [0,0,0,0,1,1,1,1] and subsets: [0,1,2,0,1,2,0,1], this means that
# # after splitting we end up with the following sets of labels: [0,0,1], [0,1,1], [0,1].
# #
# # Calculate the information gain when these n elements are split up.
# #
# # The formula for individual entropy is:
# #
# # ϕ(p)=− ∑i pi log2 pi
#
# # The formula for overall (split-) entropy is:
# #
# # Im=∑j=1s Nj/N * ϕ(pj)
# # There will be s subsets (possibly more than 1).
# #
# # Return the information gain (IG = entropy - split_entropy).
# #
# # Hint: to call a function from the Solution class you can use: Solution.function()
#
#
# class Solution:
#
#     def information_gain(labels, subsets):
#         ent = Solution.entropy(labels)
#         Im = Solution.split_entropy(labels, subsets)
#         return ent - Im
#
#     def entropy(labels):
#         n = len(labels)
#
#         # calculate the fraction 1s and 0s
#         total_1s = labels.sum()
#         frac_1s = total_1s / n
#         total_0s = n - total_1s
#         frac_0s = total_0s / n
#
#         # get individual entropies
#         # entropy_1s
#         entropy_1s = 0
#         if frac_1s > 0:
#             entropy_1s = - frac_1s * log2(frac_1s)
#         # entropy_0s
#         entropy_0s = 0
#         if frac_0s > 0:
#             entropy_0s = - frac_0s * log2(frac_0s)
#
#         # get total entropy
#         return entropy_1s + entropy_0s
#
#     def split_entropy(labels, subsets):
#
#         n = len(labels)
#         information = 0
#         unique_subsets = np.unique(subsets)
#
#         for i in unique_subsets:
#             subset_labels = labels[subsets == i]
#             information += len(subset_labels) / n * Solution.entropy(subset_labels)
#
#         return information
