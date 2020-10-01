from exercise5_3 import *

# n_repetitions = 10
# max_neighbours = 20
# accuracies = np.zeros((max_neighbours, n_repetitions))
# mean_accuracies = np.zeros(max_neighbours)
# seeds = [x for x in range(n_repetitions)]
#
# for i in range(n_repetitions):
#     # Generate a new split of train and testset
#     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=seeds[i])
#     for k in range(1, max_neighbours + 1):
#         # START ANSWER
#         predictions = predict(X_train, X_test, Y_train, Y_test, k)
#         accuracies[k-1][i] = accuracy_score_self(Y_test, predictions)
#
# for i in range(max_neighbours):
#     mean_accuracies[i] = np.sum(accuracies[i]) / n_repetitions
#         #accuracy for k=1 with 10 repetitions split training data randomly, get  average accuracy per row
#         #accuracy for k=2 with 10 repetitions split training data randomly, get average accuracy per row
#         #accuracy for k=3 with 10 repetitions split training data randomly, get average accuracy per row
#         #...
#         # accuracy for k=20 with 10 repetitions split training data randomly, get average accuracy per row
#         # total 200 accuracies
#         # compare avg accuracies, the highest average accuracy is the best choice for k parameter.
#         # we optimized for k parameter
#         # This was k-fold cross validation
#
#         # END ANSWER
#
# plt.plot(range(1, 21), mean_accuracies)
# plt.title('The averaged accuracies over the different values of k')
# plt.xlabel('k')
# plt.ylabel('Accuracy');
# plt.show()