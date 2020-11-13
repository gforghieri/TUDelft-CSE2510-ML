import numpy as np

# Given that a p-dimensional Gaussian pdf is given by:

# p(x)= 1 / sqrt( ((2π)^p) * det(Σ)) * exp( (−1 / 2) * (x−μ)T * Σ−1 * (x−μ) )

# and given a 1-dimensional trainingset with n=4 datapoints:
#     x1=1.2, x2=3.4, x3=2.7, x4=4.5.
# Implement this probability density function p(x) in Python where the parameters μ and Σ are estimated from the given training set.
# The function should output the value of p(x).


class Solution():

    def solution(testx):
        traindata = [ 1.2, 3.4, 2.7, 4.5]

        mu = np.mean(traindata)
        var = np.var(traindata)
        px = np.exp(-0.5*(testx-mu)**2/var)/np.sqrt(2.*np.pi*var)
        return px
