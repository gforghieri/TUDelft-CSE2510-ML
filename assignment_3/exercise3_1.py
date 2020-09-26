from exercise1_1 import *
from exercise2_1 import *


def get_neighbours(training_set, test_instance, k):
    """
    Calculate distances from test_instance to all training points.
    :param training_set: [n x d] numpy array of training samples (n: number of samples, d: number of dimensions).
    :param test_instance: [d x 1] numpy array of test instance features.
    :param k: number of neighbours to return.
    :return: list of length k with neighbour indices, with increasing distance of the neighbours
    """
    neighbours_indices = []
    # START ANSWER
    all_distances = []
    for i in range(len(training_set)):
        all_distances = np.append(all_distances, (euclidean(training_set[i], test_instance)))

    neighbours_indices = np.argsort(all_distances)[:k]

    # END ANSWER
    return neighbours_indices


neighbours = get_neighbours(X_train, X_test[0], 5)

# check whether your algorithm is correct
print('The indices returned by your algorithm are:', neighbours)

# assert neighbours == [63, 41, 76, 51, 10]
