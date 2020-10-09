from exercise4_3 import *

from sklearn.model_selection import train_test_split

# flatten de data so all items are 1D and append an extra one feature to every item
binary_digits_pixels = add_one_features(binary_digits_images.reshape(binary_digits_images.shape[0], -1))

# the shape should be (360, 65)
print(binary_digits_pixels.shape)

# Split dataset into train and test set
x_train_digits, x_test_digits, y_train_digits, y_test_digits = train_test_split(binary_digits_pixels,
                                                                                binary_digits_labels, test_size=0.3)

# train a theta vector for the features and labels of the binary digits:
theta_digits = train_theta(x_train_digits, y_train_digits)
print("theta vector: " + str(theta_digits))


def predict_binary(x_test, theta):
    """
    Predicts a label for each image in x_test using theta.
    :param x_test: an array of size (n, 65) of all test images.
    :param theta: a (65,) array of trained theta.
    :return: an integer array of size (n,) of labels for each test_image.
    """
    predictions = np.zeros(x_test.shape[0], dtype=int)
    # START ANSWER
    predictions = np.where(hypothesis(x_test, theta) > 0.5, 1, 0)
    # END ANSWER
    return predictions


x_test = np.array([[1, 2, 3, 1], [-1, 2, 1.5, 1], [4, -5, 2, 4]])
theta = np.array([1, -1, 2, -2])
predictions = predict_binary(x_test, theta)
print(predictions)

assert (predictions == np.array([1, 0, 1])).all()
assert predictions.dtype == np.dtype('int')
