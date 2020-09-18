# Given an array A with shape (N,) make an array with all elements of A in reverse order
# and return as a matrix of size (N, 1).
import numpy as np

A = np.arange(6)

def reverse(A):
    result = None
    # START ANSWER
    result = np.flip(A).reshape(len(A), 1)
    # END ANSWER
    return result

rev = reverse(A)
assert (rev == np.array([[5],[4],[3],[2],[1],[0]])).all()
rev