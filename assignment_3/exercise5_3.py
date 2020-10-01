from exercise3_2 import *
from exercise5_2 import *


def plot_errors(X_train, X_test, Y_train, Y_test, predictions, k):
    """
    Plots the test points that were misclassified and their nearest neighbours using plot_neighbours.
    """
    misclassified_indices = []
    neighbours_of_missclassified_indices = []

    # START ANSWER
    for i in range(len(predictions)):
        if predictions[i] != Y_test[i]:
            misclassified_indices.append(i)
            neighbours_of_missclassified_indices = np.append(neighbours_of_missclassified_indices,
                                                             get_neighbours(X_train, X_test[i], k=5))

    for i in range(len(misclassified_indices)):
        test_instance = X_test[misclassified_indices[i]]
        k = 5
        plt.title('Test instance %s and its nearest neighbors' % (misclassified_indices[i] + 1))
        plot_neighbours(X_train, Y_train, test_instance, k)
    return


# plot_errors(X_train, X_test, Y_train, Y_test, predictions, k)
