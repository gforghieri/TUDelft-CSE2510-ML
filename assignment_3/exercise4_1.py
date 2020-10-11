from collections import Counter  # to count unique occurances of items in array, for majority voting

from exercise3_1 import *


def get_majority_vote(neighbour_indices, training_labels):
    """
    Given an array of nearest neighbours indices for a given test case,
    tally up their classes to vote on the correct class for the test instance.
    :param neighbours: list of nearest neighbour indices.
    :param training_labels: the list of labels for each training instance.
    :return: the label of most common class.
    """
    most_common = -1
    # START ANSWER
    neighbour_indices = np.array(neighbour_indices)
    training_labels = np.array(training_labels)

    labels_of_neighbours = training_labels[neighbour_indices]
    c = Counter(labels_of_neighbours)
    most_common = c.most_common()

    if (len(most_common) > 1 and most_common[0][1] == most_common[1][1]):
        print("TODO:")
        most_common = most_common[0][0]
    else:
        most_common = most_common[0][0]
        # END ANSWER
    return most_common


predicted_label = get_majority_vote(neighbours, Y_train)
print('Your predicted label:', predicted_label)

assert predicted_label == 0
assert get_majority_vote([0, 1, 2, 3, 4], [3, 1, 1, 3, 0]) == 3
