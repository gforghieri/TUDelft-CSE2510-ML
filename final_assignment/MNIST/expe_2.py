from sklearn.utils.validation import check_is_fitted

from expe_1 import *

random_state = 42

models = {
    "GaussianNB": GaussianNB(),
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=random_state),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3, weights="distance"),
    # START ANSWER
    # 𝐄𝐱𝐞𝐫𝐜𝐢𝐬𝐞 1  Extent the list of models with theSVC and LogisticRegression algorithms.
    # Give the SVM a poly kernel. Also, give both algorithms a regularization constant C=0.5 and random_state=42.
    "SVM": SVC(kernel='poly', C=10, random_state=42),
    "LogisticRegression": LogisticRegression(C=10, random_state=42, penalty='none')
    # END ANSWER
}

assert "GaussianNB" in models and isinstance(models["GaussianNB"], GaussianNB), "There is no GaussianNB in models"
assert "DecisionTreeClassifier" in models and isinstance(models["DecisionTreeClassifier"],
                                                         DecisionTreeClassifier), "There is no DecisionTreeClassifier in models"
assert "KNeighborsClassifier" in models and isinstance(models["KNeighborsClassifier"],
                                                       KNeighborsClassifier), "There is no KNeighborsClassifier in models"
assert "SVM" in models and isinstance(models["SVM"], SVC), "There is no SVC in models"
assert "LogisticRegression" in models and isinstance(models["LogisticRegression"],
                                                     LogisticRegression), "There is no LogisticRegression in models"

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

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # data to plot
    # accuracies_28x28
    # accuracies_8x8
    #
    # # create plot
    # fig, ax = plt.subplots()
    # bar_width = 0.35
    # X = np.arange(len(models))
    #
    # p1 = plt.bar(X, accuracies_28x28, bar_width, color='b',
    #              label='28x28 dataset')
    #
    # # The bar of second plot starts where the first bar ends
    # p2 = plt.bar(X + bar_width, accuracies_8x8, bar_width,
    #              color='g',
    #              label='8x8 dataset')
    #
    # plt.xlabel(models.keys())
    # plt.ylabel('Accuracy Scores')
    # plt.title('Scores in each subject')
    # plt.xticks(X + (bar_width / 2), (
    # 'GaussianNB', 'DummyClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'SVM', 'LogisticRegression'))
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # # Pass the x and y cordinates of the bars to the
    # # function. The label argument gives a label to the data.
    # plt.bar(name, accuracy, label="Data 1")
    # plt.legend()
    #
    # # The following commands add labels to our figure.
    # plt.xlabel(name + "8x8")
    # plt.ylabel('Accuracy score')
    # plt.title('Performance of classifiers on 8x8 vs 28x28 MNIST dataset')
    #
    # plt.show()

import numpy as np
import matplotlib.pyplot as plt

# data to plot

accuracies_28x28

# marks_john = [0.1, 0.5, 0.8, 0.1, 0.5, 0.9]
# marks_sam = [0.25, 0.5, 0.5, 0.5, 0.5, 0.1]



# create plot
fig, ax = plt.subplots()
bar_width = 0.35
X = np.arange(6)

p1 = plt.bar(X, accuracies_28x28 , bar_width, color='b',
             label='28x28 dataset')

# The bar of second plot starts where the first bar ends
p2 = plt.bar(X + bar_width, accuracies_8x8, bar_width,
             color='g',
             label='8x8 dataset')

plt.xlabel('Algorithms')
plt.ylabel('Accuracy Scores')
plt.title('Performance of different algorithms per dataset 28x28 vs 8x8')
plt.xticks(X + (bar_width / 2), (
'GaussianNB', 'Dummy', 'DecisionTree', 'KNN', 'SVM', 'LogisticR'))
plt.legend()

plt.tight_layout()
plt.show()
