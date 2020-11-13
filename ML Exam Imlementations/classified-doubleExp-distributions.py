import numpy as np


# Assume we have a one-dimensional classification problem with two classes.
# Further assume that both classes have a double exponential distribution:
#     p(x)=(1/2β) * exp(− ∣ (x − μ / β) ∣ )
#
# where μ is the location parameter, and β is the scale parameter.
# For class 1, we have that μ1=0,β1=1 and for class 2, we have that μ2=3β2=2.
# Assuming that both classes are equally likely, implement a function that computes the posterior probability of class 1.

class Solution():

    def solution(testx):
        mu1 = 0.0
        mu2 = 3.0
        beta1 = 1.0
        beta2 = 2.0

        px_giveny1 = np.exp(-np.abs((testx-mu1)/beta1))/(2.*beta1)
        px_giveny2 = np.exp(-np.abs((testx-mu2)/beta2))/(2.*beta2)
        py1 = 0.5
        py2 = 0.5

        py1_given_x = px_giveny1*py1/(px_giveny1*py1 + px_giveny2*py2)

        return py1_given_x
