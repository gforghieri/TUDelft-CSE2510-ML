import numpy as np
from library import hypothesis, apply_gradient, predict_binary, compute_accuracy

# Given are a multi-class dataset and their labels, a binary logistic regression classifier, and accuracy function. Implement one vs. all method to train the classifier and report the labels of the testset.
#
# Hint:
# First, use the training set to optimize theta using the hypothesis and apply gradient function. Afterwards, you can use this theta in the predict_binary function.
#
# The predict_binary function returns the probability that an instance belongs to a class. Remember how you can use this probability during one vs. all classification.

class Solution():

    def solution(x_train, y_train, x_test):

        # Different labels
        labels = np.unique(y_train)
        # Learning rate
        alpha = 0.1
        # Array where to store the probability for all test instances for all classes
        y_test_probabilities = np.zeros([len(x_test),len(labels)])

        # Loop over the binary classifiers
        for j, label in enumerate(labels):

            # Initialize theta as zero
            theta = np.zeros([np.shape(x_train)[1],])
            # Number of epochs
            n_epochs = 5

            # Find optimal theta
            # STUDENT
            for epoch in range(n_epochs):
                for i in range(len(x_train)):
                    if y_train[i] == label:
                        theta = apply_gradient(theta, x_train[i], 1, alpha)
                    else:
                        theta = apply_gradient(theta, x_train[i], 0, alpha)


            # Predict probability of all test set instances
            y_test_probabilities[:,j] = predict_binary(x_test, theta)

        # Find label of the test instances
        # STUDENT
        y_predictions = labels[np.argmax(y_test_probabilities, axis = 1)]

        return y_predictions
