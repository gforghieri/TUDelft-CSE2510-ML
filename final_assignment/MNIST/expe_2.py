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
assert "DecisionTreeClassifier" in models and isinstance(models["DecisionTreeClassifier"], DecisionTreeClassifier), "There is no DecisionTreeClassifier in models"
assert "KNeighborsClassifier" in models and isinstance(models["KNeighborsClassifier"], KNeighborsClassifier), "There is no KNeighborsClassifier in models"
assert "SVM" in models and isinstance(models["SVM"], SVC), "There is no SVC in models"
assert "LogisticRegression" in models and isinstance(models["LogisticRegression"], LogisticRegression), "There is no LogisticRegression in models"

# 4. fit
# 5. predict
# 6. evaluate
# 7. graph/plot
# 8. analyze

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
    print(name + "8x8")
    print("- accuracy_score", accuracy)
    print("- f1_score", f1_score_value)
    plt.hist(accuracy_score, label=name + "8x8")
    plt.plot(accuracy_score, label=name + "8x8")

    # plt.hist(setosa_X_train[:, feature_idx], label=iris.target_names[0])
    # plt.hist(versicolor_X_train[:, feature_idx], label=iris.target_names[1])
    # plt.hist(virginica_X_train[:, feature_idx], label=iris.target_names[2])
    #
    # # Plot your PDFs here
    # xs = np.linspace(0, 7, 100)
    #
    # # START ANSWER
    # plt.plot(xs, normal_PDF(xs, mean_setosa, sd_setosa))
    # plt.plot(xs, normal_PDF(xs, mean_versicolor, sd_versicolor))
    # plt.plot(xs, normal_PDF(xs, mean_virginica, sd_virginica))
    #
    # # END ANSWER
    #
    # plt.xlabel(iris.feature_names[feature_idx])
    # plt.ylabel('Number of flowers / PDF')
    # plt.legend()
    # plt.show()