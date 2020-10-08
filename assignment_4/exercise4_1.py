from exercise3_2 import *

near_0 = 1e-16
near_1 = 1.0 - near_0


def log_likelihood(h_x, y):
    """
    Computes the log likelihood of your classifier.
    :param h_x: numpy array of predicted probabilities.
    :param y: numpy array of actual labels (positive (1) or negative (0)).
    :return: The log likelihood.
    """
    log_likelihood = 0
    # START ANSWER

    # There might be warnings from numpy regarding division by zero and invalid value.
    # You can solve this by replacing 0/1 values with near_0,near_1 values with the np.where function
    h_x = np.where(h_x == 0, near_0, np.where(h_x == 1, near_1, h_x))

    log_likelihood = np.sum((y * np.log(h_x)) + ((1 - y) * np.log(1 - h_x)))

    # END ANSWER
    return log_likelihood


# These predictions should do very well
h_x1 = np.array([0.01, 0.01, 0.99, 0.99])
y1 = np.array([0, 0, 1, 1])
ll1 = log_likelihood(h_x1, y1)
print(ll1)

# These predictions should do ok
h_x2 = np.array([0.2, 0.1, 0.9, 0.8])
y2 = np.array([0, 0, 1, 1])
ll2 = log_likelihood(h_x2, y2)
print(ll2)

# These predictions should do bad
h_x3 = np.array([0.9, 0.8, 0.99, 0.3, 0.1])
y3 = np.array([0, 0, 1, 1, 1])
ll3 = log_likelihood(h_x3, y3)
print(ll3)

assert np.isclose(ll1, -0.040201)
assert np.isclose(ll2, -0.657008)
assert np.isclose(ll3, -7.428631)

# There might be warnings from numpy regarding division by zero and invalid value.
# You can solve this by replacing 0/1 values with near_0,near_1 values with the np.where function
h_x4 = np.array([0.0, 0.1, 1.0, 0.95])
y4 = np.array([0, 0, 1, 1])
ll4 = log_likelihood(h_x4, y4)
print(ll4)

h_x5 = np.array([1.0, 0.99, 0.0, 0.01])
y5 = np.array([0, 0, 1, 1])
ll5 = log_likelihood(h_x5, y5)
print(ll5)

assert ll5 < -10.0
assert np.isclose(ll4, -0.156653, rtol=0.5)
# Due to the wrong predictions, this likelihood is very low

# Use only the width feature
width_features = binary_digits_features[:,0]
width_features_prime = add_one_features(width_features)
binary_digits_labels

# axis limits to plot
min_theta_0 = -20.0
max_theta_0 = 5.0
min_bias = -30.0
max_bias = 150.0

# resolution for both axis
N = 50
thetas_0 = np.linspace(min_theta_0, max_theta_0, N)
biases = np.linspace(min_bias, max_bias, N)

# 2D array with log likelihoods to be filled
log_likelihoods = np.zeros(shape=(len(biases), len(thetas_0)))
# fill the 2D array
for i_theta_0, theta_0 in enumerate(thetas_0):
    for i_bias, bias in enumerate(biases):
        # construct theta
        theta = np.array([theta_0, bias])
        h_x = hypothesis(width_features_prime, theta)
        ll = log_likelihood(h_x, binary_digits_labels)
        log_likelihoods[i_bias,i_theta_0] = ll

# plot log likelihoods
X, Y = np.meshgrid(thetas_0, biases)
cs = plt.contourf(X, Y, log_likelihoods, cmap="PuRd")
plt.title('Log-Likelihoods')
plt.xlabel(r'$\theta_0$')
plt.ylabel('bias')
plt.colorbar(cs)

plt.show()