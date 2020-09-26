from exercise1_1 import *

from scipy.spatial import distance


def euclidean(p, q):
    """
    Computes the euclidean distance between point p and q.
    :param p: point p as a numpy array.
    :param q: point q as a numpy array.
    :return: distance as float.
    """
    dist = 0
    # START ANSWER
    dist = np.sqrt(np.dot((p - q), (p - q)))
    # END ANSWER
    return dist


# check whether your algorithm is correct
a = np.array([2, 4, 8])
b = np.array([3, 5, 9])

print('The output of your algorithm:', euclidean(a, b))
assert np.isclose(euclidean(a, b), distance.euclidean(a, b))

# Other distances in the course so far, manhattan distance, hamming distance, eucledian distance