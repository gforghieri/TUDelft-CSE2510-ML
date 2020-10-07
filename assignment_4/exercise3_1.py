from exercise2_1 import *


# z = np.linspace(-5, 5, 100)
# sigmoid = 1 / (1 + np.exp(-z))
# plt.title('Sigmoid function')
# plt.xlabel(r'$\theta^T x + bias$')
# plt.ylabel(r'$\sigma(\theta^T x + bias$)')
# plt.plot(z, sigmoid)

def plot_hypothesis(features, labels, theta, bias):
    widths = features[:, 0]
    # some noise is added to better visualize the labels of the datapoints
    labels_noise = labels + np.random.normal(0, .05, labels.shape)

    plt.scatter(widths, labels_noise, c='red')

    x = np.linspace(np.min(widths), np.max(widths), 100)
    sigmoid_1D = 1 / (1 + np.exp(-(theta * x + bias)))
    plt.title('hypothesis function')
    plt.xlabel('x')
    plt.ylabel('label (red)/probability class 1 (blue)')
    plt.plot(x, sigmoid_1D)
    plt.show()


# try to find proper values for theta and bias
# such that the sigmoid properly goes through both the datapoints with label 0 and 1
theta = 0
bias = 0
# START ANSWER
theta = 0.6
bias = -5
# END ANSWER

plot_hypothesis(binary_digits_features, binary_digits_labels, theta, bias)


# This function adds an extra 1.0 to every feature vector
def add_one_features(data):
    return np.vstack((data.T, np.ones(len(data)))).T


binary_digits_features_prime = add_one_features(binary_digits_features)
print(binary_digits_features_prime[:10])
