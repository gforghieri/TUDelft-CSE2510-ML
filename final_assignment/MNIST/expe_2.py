from sklearn.utils.validation import check_is_fitted

from expe_1 import *

# 4. fit
# 5. predict
# 6. evaluate
# 7. graph/plot
# 8. analyze


accuracies_28x28 = []
accuracies_8x8 = []

# 4. fit 28x28
for name, model in models.items():
    # START ANSWER
    model.fit(scaled_X_train_28x28, y_train)
    # END ANSWER

for model in models.values():
    check_is_fitted(model)

from sklearn.metrics import f1_score, accuracy_score, log_loss

# 5. predict 28x28 and # 6. evaluate 28x28
for name, model in models.items():
    predictions_28x28 = model.predict(scaled_X_test_28x28)
    f1_score_value = f1_score(y_true=y_test, y_pred=predictions_28x28, average="weighted")
    accuracy = accuracy_score(y_true=y_test, y_pred=predictions_28x28)
    accuracies_28x28 = np.append(accuracies_28x28, accuracy)
    # logloss = log_loss(y_true=y_test, y_pred=model.predict_proba(scaled_X_test_28x28))
    print(name + "28x28")
    print("- accuracy_score", accuracy)
    print("- f1_score", f1_score_value)
    # print("- logloss", logloss)

# 4. fit 8x8
for name, model in models.items():
    # START ANSWER
    model.fit(scaled_X_train_8x8, y_train)
    # END ANSWER

for model in models.values():
    check_is_fitted(model)

from sklearn.metrics import f1_score, accuracy_score

# 5. predict 8x8 and # 6. evaluate 8x8
for name, model in models.items():
    predictions_8x8 = model.predict(scaled_X_test_8x8)
    f1_score_value = f1_score(y_true=y_test, y_pred=predictions_8x8, average="weighted")
    accuracy = accuracy_score(y_true=y_test, y_pred=predictions_8x8)
    accuracies_8x8 = np.append(accuracies_8x8, accuracy)
    print(name + "8x8")
    print("- accuracy_score", accuracy)
    print("- f1_score", f1_score_value)

# create plot
fig, ax = plt.subplots()
bar_width = 0.35
X = np.arange(6)

p1 = plt.bar(X, accuracies_28x28, bar_width, color='b',
             label='28x28 dataset')

# The bar of second plot starts where the first bar ends
p2 = plt.bar(X + bar_width, accuracies_8x8, bar_width,
             color='g',
             label='8x8 dataset')

plt.xlabel('Algorithms')
plt.ylabel('Accuracy Scores')
plt.title('Performance of different algorithms on the 28x28 vs 8x8 datasets')
plt.xticks(X + (bar_width / 2), (
    'GaussianNB', 'Dummy', 'DecisionTree', 'KNN', 'SVM', 'LogisticR'))
plt.legend()

plt.tight_layout()
plt.show()
