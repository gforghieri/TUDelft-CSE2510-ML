from sklearn.metrics import accuracy_score
from exercise3_1 import *
from exercise4_1 import *


def predict(X_train, X_test, Y_train, Y_test, k=5):
    """
    Predicts all labels for the test set, using k-nn on the training set and computes the accuracy.
    :param X_train: the training set features.
    :param X_test: the test set features.
    :param y_train: the training set labels.
    :param y_test: the test set labels.
    :return: list of predictions.
    """

    # generate predictions
    predictions = []
    # For each instance in the test set, get nearest neighbours and majority vote on predicted class
    # START ANSWER

    all_neighbours_groups = []

    for i in range(len(X_test)):
        all_neighbours_groups += [(get_neighbours(X_train, X_test[i], k))]

    for i in range(len(all_neighbours_groups)):
        predictions += [(get_majority_vote(all_neighbours_groups[i], Y_train))]
        # predictions = np.append(predictions, get_majority_vote(all_neighbours_groups[i], Y_train))

        # END ANSWER
    return predictions


k = 5
predictions = predict(X_train, X_test, Y_train, Y_test, k)

# summarise performance of the classification using scikit-learn
accuracy = accuracy_score(Y_test, predictions)
print('The overall accuracy of the model using scikit-learn is:', accuracy)

assert predictions == [0, 1, 1, 2, 1, 1, 2, 0, 2, 0, 2, 1, 2, 0, 0, 2, 0, 1, 2, 1, 1, 2, 2, 0, 2, 1, 1, 0, 2, 2, 1, 1,
                       0, 0, 0, 1, 1, 0, 1, 2, 1, 2, 0, 1, 1, 0, 0, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1]
assert np.isclose(accuracy, 0.9666666666666667)
