import numpy as np

# Hard combination
#
# Use the hard type combination rule below to combine the output of classifiers.
#
# Input: a numpy array containing the predicted labels of L classifiers for a test set of size n (size: [n x L])
#
# Output: numpy array containing the predicted label for each instance in the test set (size: [n x 1]).
# If an instance is rejected, return None for that item.
#
# Hard combination rule:
# l= | L/2 + 1 |
# Where:
# L = number of classifiers
# l = minimum number of classifiers that should agree on the class label

class Solution():

    def solution(predicted_labels):

        L = np.shape(predicted_labels)[1]
        lc = np.floor(L/2 + 1)

        combined_labels = []

        for item in predicted_labels:
            labels, counts = np.unique(item, return_counts = True)
            if np.max(counts) >= lc:
                newlabel = labels[np.argmax(counts)]
            else:
                newlabel = None
            combined_labels.append(newlabel)

        return combined_labels
