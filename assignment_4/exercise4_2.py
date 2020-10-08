from exercise4_1 import *


def calculate_gradients(theta, x, y):
    """
    Calculate the gradient for every datapoint in x
    :param theta: numpy array of theta
    :param x: numpy array of the features
    :param y: the label (positive (1) or negative (0))
    :return: The gradients for every datapoint in x
    """
    gradients = np.zeros((len(x), len(theta)))
    # START ANSWER
    temp = (y-hypothesis(x, theta))
    for i in range(len(x)):
        gradients[i] = temp[i] * x[i]
    return gradients


theta = np.array([1, 1.5, 2.5])
x = np.array([[-10, 5, 1], [0.5, 1, 1]])
y = np.array([0, 1])
gradients = calculate_gradients(theta, x, y)
print(gradients)

assert np.isclose(gradients[0], np.array([5.0, -2.5, -0.5])).all()
assert np.isclose(gradients[1], np.array([0.00549347, 0.01098694, 0.01098694]), atol=0.0001).all()
