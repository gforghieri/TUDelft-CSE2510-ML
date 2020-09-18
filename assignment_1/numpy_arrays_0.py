# Given two arrays A and B each of the same size calculate their sum (elementwise) and their product (elementwise).
import numpy as np

A = np.arange(5)
B = np.arange(5, 10)

def sum_arrays(A, B):
    result = None
    # START ANSWER
    result = A + B
    # END ANSWER
    return result


def multiply_arrays(A, B):
    result = None
    # START ANSWER
    result = A * B
    # END ANSWER
    return result


sum_AB = sum_arrays(A, B)
assert (sum_AB == np.array([5, 7, 9, 11, 13])).all()

mult_AB = multiply_arrays(A, B)
assert (mult_AB == np.array([0, 6, 14, 24, 36])).all()

sum_AB, mult_AB
