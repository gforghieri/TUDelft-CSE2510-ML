from exercise3_1 import *

# implement the hypothesis function so that it works for thetas/features of arbitrary length
def hypothesis(x, theta):
    """
    Calculate the hypothesis function for every datapoint in x
    :param x: numpy array of size (n, d) where n is the number of samples
    and d is the number of features per sample including the 1 extra feature
    :param theta: numpy array of size (d,)
    :return: predicted probability.
    """
    # START ANSWER
    sigmoid = 1 / (1 + np.exp(-np.dot(x, theta)))
    # END ANSWER
    return sigmoid

x = binary_digits_features_prime

# To test our hypothesis function, we set three different theta vectors
# All 1
theta_ones = np.ones(3)
# All 0
theta_zeros = np.zeros(3)
# All -1
theta_min_ones = -5 * np.ones(3)

# And apply the prediction
hypothesis_ones = hypothesis(x, theta_ones)
hypothesis_zeros = hypothesis(x, theta_zeros)
hypothesis_min_fives = hypothesis(x, theta_min_ones)

# Output for each theta vector
# expected = 1.0
print("Prediction ones: {}".format(hypothesis_ones[:5]))
# expected = 0.5
print("Prediction zeros: {}".format(hypothesis_zeros[:5]))
# expected = ~0
print("Prediction minus fives: {}".format(hypothesis_min_fives[:5]))

assert np.isclose(hypothesis_zeros, 0.5).all()
assert np.isclose(hypothesis_min_fives, 0).all()
assert np.isclose(hypothesis_ones, 1).all()
