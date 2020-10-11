from exercise6_1 import *


def predict_multiclass(x_test, theta):
    """
    Predicts a label for each image in x_test using theta.
    :param x_test: an array of size (n, 65) of all test images.
    :param theta: an (10,65) array of trained thetas.
    :return: an array of size (n,) of labels for each test_image.
    """
    predictions = np.zeros(x_test.shape[0], dtype=int)
    for i, x in enumerate(x_test):
        # START ANSWER
        temp_predictions = np.zeros(10)
        for j in range(len(multiclass_thetas)):
            temp_predictions[j] = hypothesis(x, multiclass_thetas[j])
        predictions[i] = np.argmax(temp_predictions)
        # END ANSWER
    return predictions


predictions = predict_multiclass(x_test_digits, multiclass_thetas)
# And print the accuracy
accuracy = compute_accuracy(predictions, y_test_digits)
print("accuracy: " + str(accuracy))

assert accuracy > 0.9

for i in range(n_classes):
    plot_theta_image(multiclass_thetas[i], str(i))
