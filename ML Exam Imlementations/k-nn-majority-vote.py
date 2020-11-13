import numpy as np

# Given an array of nearest neighbors and the list of labels for each instance in the array of the nearest neighbours,
# implement a function that performs majority vote. Return the label of the most common class.
# If it is a tie (i.e. multiple classes are equally common), your function should only return the first class encountered.
#
# Input:
# - x: numpy array with the indices of nearest neighbors
# - y: numpy array of labels
#
# Output:
# - label of the most common class

class Solution():

    def nearest_neighbors(x,y):

        # find labels of the neighbors
        labels_neighbors = y[x]

        maxcount = 0
        most_common_label = -1
        # Iterate over all possible labels
        for current_label in labels_neighbors:
            count = 0
            # Iterate over all neighbors
            for neighbor_label in labels_neighbors:
                if neighbor_label == current_label:
                    count += 1
            if count > maxcount:
                maxcount = count
                most_common_label = current_label

        return most_common_label
