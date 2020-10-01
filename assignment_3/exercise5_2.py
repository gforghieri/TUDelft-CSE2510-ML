from exercise5_1 import *


def accuracy_score_self(Y_test, predictions):
    """
    Computes the accuracy of a test set as the fraction of items that was classified correctly.
    :param y_test: the list of true labels for the test set.
    :param y_pred: the list of predicted labels for the test set.
    :return: accuracy as a floating point.
    """

    accuracy = 0
    # START ANSWER
    correct_labels = []

    for i in range(len(predictions)):
        if predictions[i] == Y_test[i]:
            correct_labels = np.append(correct_labels, predictions[i])
        else:
            print(i)
            print(Y_test[i])
            print(predictions[i])

    accuracy = len(correct_labels) / len(predictions)

    # END ANSWER
    return accuracy


# summarise performance of the classification
accuracy_self = accuracy_score_self(Y_test, predictions)
print('The overall accuracy of the model using your implementation of accuracy:', accuracy_self)
assert np.isclose(accuracy, accuracy_self)
