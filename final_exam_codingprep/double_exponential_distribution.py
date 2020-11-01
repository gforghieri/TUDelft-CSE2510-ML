import numpy as np


class Solution():

    def solution(testx):
        mu1 = 0.0
        mu2 = 3.0
        beta1 = 1.0
        beta2 = 2.0

        px_giveny1 = (1 / (2 * beta1)) * np.exp(((-np.abs((testx - mu1) / beta1))))
        px_giveny2 = (1 / (2 * beta2)) * np.exp(((-np.abs((testx - mu2) / beta2))))

        py1 = 0.5
        py2 = 0.5

        py1_given_x = px_giveny1 * py1 / (px_giveny1 * py1 + px_giveny2 * py2)

        return py1_given_x
