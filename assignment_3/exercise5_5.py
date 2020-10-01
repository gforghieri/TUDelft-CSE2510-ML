from exercise5_4 import *

k = 8

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=seed)

total_samples = X_train.shape[0]
# set up array to store accuracies
accuracies = np.zeros(total_samples + 1)

# we want to learn with at least k samples and up to the size of the train set
for i in range(k, total_samples):
    predictions = predict(X_train[:i], X_test, Y_train[:i], Y_test, k)
    accuracies[i + 1] = accuracy_score(Y_test, predictions)

# plot learning curve
plt.plot(range(total_samples + 1), accuracies)
plt.title('The learning curve for the k-NN classifier')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()