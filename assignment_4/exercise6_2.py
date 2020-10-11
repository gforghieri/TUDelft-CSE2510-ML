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


        # return hypothesis function for x is label 0
        # return hypothesis function for x is label 1
        # return hypothesis function for x is label 2
        # return hypothesis function for x is label 3
        # return hypothesis function for x is label 4
        # return hypothesis function for x is label 5
        # return hypothesis function for x is label 6
        # return hypothesis function for x is label 7
        # return hypothesis function for x is label 8
        # return hypothesis function for x is label 9
        # temp_predictions = for each image in 540 x_test images an array of (10,) returned with predictions
        # predictions = np.argmax(temp_predictions) will return the index of the label which is most likely the correct one
        # thetas help assigning weight to each of the pixels, so that certain pixels matter more than others.
        # predictions will be (540,)

        # END ANSWER
    return predictions

predictions = predict_multiclass(x_test_digits, multiclass_thetas)
# And print the accuracy
accuracy = compute_accuracy(predictions, y_test_digits)
print("accuracy: " + str(accuracy))

assert accuracy > 0.9


for i in range(n_classes):
    plot_theta_image(multiclass_thetas[i], str(i))