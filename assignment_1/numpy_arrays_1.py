# Given an array A with shape (128,) calculate the mean of the elements at even indexes.
import numpy as np

A = np.arange(128)


def mean_even_idx(A):
    result = None
    # START ANSWER
    result = np.sum(np.where(A % 2 == 0)) / (len(A)/2)
    # END ANSWER
    return result


mean_A = mean_even_idx(A)
assert mean_A == 63.0

mean_A
