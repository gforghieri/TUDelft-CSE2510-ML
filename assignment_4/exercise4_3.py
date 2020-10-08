from exercise4_2 import *


def train_theta(features, labels, n_epochs=200, theta=None, alpha=0.1):
    assert len(features) == len(labels)

    num_features = len(features[0])
    num_items = len(features)
    # Set theta to intial random values
    # Initialize theta randomly if it's not provided
    if theta is None:
        theta = np.random.normal(0, .05, num_features)

    # We go through the entire training set a number of times
    # Each of these iterations is called an epoch
    for epoch in range(n_epochs):
        # calculate the average gradient for all items and apply gradient ascent to theta
        # START ANSWER
    
        # END ANSWER

    return theta


# train a theta vector for the features and labels of the binary digits:
theta = train_theta(binary_digits_features_prime, binary_digits_labels, n_epochs=100000, alpha=0.05)

print("theta vector:  " + str(theta))
print("log likelihood: " + str(log_likelihood(hypothesis(binary_digits_features_prime, theta), binary_digits_labels)))