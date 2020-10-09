from exercise4_2 import *


def train_theta(features, labels, n_epochs=200, theta=None, alpha=0.1):
    assert len(features) == len(labels)

    num_features = len(features[0])
    num_items = len(features)
    # Set theta to intial random values
    # Initialize theta randomly if it's not provided
    if theta is None:
        theta = np.random.normal(0, 0.05, num_features)

    # # We go through the entire training set a number of times
    # # Each of these iterations is called an epoch
    for epoch in range(n_epochs):
        # calculate the average gradient for all items and apply gradient ascent to theta
        # START ANSWER
        gradients = calculate_gradients(theta, features, labels)  # shape (360,3)
        avg_gradients = np.average(gradients, axis=0)  # shape(3,)
        theta = apply_gradient(theta, avg_gradients, alpha)
        # likelihood = log_likelihood(hypothesis(binary_digits_features_prime, theta), binary_digits_labels)
        # print(epoch, likelihood)
        # END ANSWER

    return theta


# # train a theta vector for the features and labels of the binary digits:
# theta = train_theta(binary_digits_features_prime, binary_digits_labels, n_epochs=10000, alpha=0.05)

# print("theta vector:  " + str(theta))
# print("log likelihood: " + str(log_likelihood(hypothesis(binary_digits_features_prime, theta), binary_digits_labels)))


# def decision_boundary(theta, plot_x):
#     return (-1 / theta[1]) * (theta[0] * plot_x + theta[2])
#
#
# def plot_decision_boundary(theta, data, labels):
#     db_x = np.array([data[:, 0].min() - 1, data[:, 0].max() + 1])
#     db_y = decision_boundary(theta, db_x)
#     plot_scatter(data, labels, db_x=db_x, db_y=db_y)
#     plt.show()
#
#
# plot_decision_boundary(theta, binary_digits_features_prime, binary_digits_labels)
