from exercise5_1 import *


def compute_accuracy(predictions, y_true):
    """
    Computes the accuracy of the predictions based on the true labels.
    :param predictions: an array of size (n,) of the computed predictions for each image.
    :param y_true: an array of size (n,) of the true labels of each image.
    :return: the accuracy of the predictions.
    """
    accuracy = -1
    # START ANSWER
    accuracy = np.sum(np.where(predictions == y_true, 1, 0)) / len(predictions)
    # END ANSWER
    return accuracy


predictions = np.array([0, 1, 1, 0, 1])
y_true = np.array([0, 1, 0, 1, 1])

accuracy = compute_accuracy(predictions, y_true)
assert accuracy == 0.6

predictions = predict_binary(x_test_digits, theta_digits)
accuracy = compute_accuracy(predictions, y_test_digits)

print("accuracy: " + str(accuracy))
assert accuracy > 0.95

def plot_theta_image(theta, title=r"$\theta$ vector"):
    # remove bias from the image
    theta_no_bias = theta[:64].reshape(8,8)
    plt.figure()
    plt.gray()
    plt.title(title)
    plt.imshow(theta_no_bias)

print(theta_digits)
plot_theta_image(theta_digits)
theta_digits
