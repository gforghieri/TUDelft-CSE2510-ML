from exercise5_2 import *

# Set learning rate (try experimenting with this)
alpha = 0.001

# Set theta to intial value of None
theta_digits = None

# We go through the entire training set a number of times
# Each of these iterations is called an epoch
n_epochs = 50

accuracies_train = []
accuracies_test = []
log_likelihoods_train = []
log_likelihoods_test = []

for epoch in range(n_epochs):
    theta_digits = train_theta(x_train_digits, y_train_digits, n_epochs=1, theta=theta_digits, alpha=alpha)
    # calculate accuracy
    accuracy_train = -1
    accuracy_test = -1
    # START ANSWER
    predictions_train = predict_binary(x_train_digits, theta_digits)
    accuracy_train = compute_accuracy(predictions_train, y_train_digits)

    predictions_test = predict_binary(x_test_digits, theta_digits)
    accuracy_test = compute_accuracy(predictions_test, y_test_digits)

    # END ANSWER
    accuracies_train.append(accuracy_train)
    accuracies_test.append(accuracy_test)

    # calculate log likelihood
    ll_train = 0
    ll_test = 0
    # START ANSWER
    ll_train = log_likelihood(hypothesis(x_train_digits, theta_digits), y_train_digits)
    ll_test = log_likelihood(hypothesis(x_test_digits, theta_digits), y_test_digits)

    # END ANSWER
    log_likelihoods_train.append(ll_train)
    log_likelihoods_test.append(ll_test)

# plt.plot(np.arange(len(accuracies_train)), accuracies_train, label='train')
# plt.plot(np.arange(len(accuracies_test)), accuracies_test, label='test')
# plt.title('accurracy')
# plt.xlabel('epoch')
# plt.ylabel('accurracy')
# plt.legend(loc=3)
# plt.show()

# plt.plot(np.arange(len(log_likelihoods_train)), log_likelihoods_train, label='train')
# plt.plot(np.arange(len(log_likelihoods_test)), log_likelihoods_test, label='test')
# plt.title('log likelihood')
# plt.xlabel('epoch')
# plt.ylabel('log likelihood')
# plt.legend(loc=3)
# plt.show()
