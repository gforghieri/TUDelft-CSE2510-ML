from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import make_scorer
from expe_1 import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, accuracy_score, log_loss, make_scorer
from sklearn.metrics import f1_score, accuracy_score

# 4. fit
# 5. predict
# 6. evaluate
# 7. graph/plot
# 8. analyze

accuracies_28x28 = []
accuracies_8x8 = []

# The first plot is done using the simpler “Train-Test Evaluation With Correct Data Preparation” method.
# 4. fit 28x28
for name, model in models.items():
    # START ANSWER
    model.fit(scaled_X_train_28x28, y_train)
    # END ANSWER

for model in models.values():
    check_is_fitted(model)

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
def create_comparison_plot(a, b):
    fig, ax = plt.subplots()

    bar_width = 0.35
    X = np.arange(6)

    p1 = plt.bar(X, a, bar_width, color='b',
                 label='28x28 dataset')

    # The bar of second plot starts where the first bar ends
    p2 = plt.bar(X + bar_width, b, bar_width,
                 color='g',
                 label='8x8 dataset')

    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy Scores')
    plt.title('Performance of different algorithms on the 28x28 vs 8x8 datasets')
    plt.xticks(X + (bar_width / 2), (
        'GaussianNB', 'Dummy', 'DecisionTree', 'KNN', 'SVM', 'LogisticR'))
    plt.legend()

    def autolabel(ps):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in ps:
            height = np.round(rect.get_height(), 2)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(p1)
    autolabel(p2)

    plt.tight_layout()
    plt.show()


create_comparison_plot(accuracies_28x28, accuracies_8x8)


# The second plot is “Cross-Validation Evaluation With Correct Data Preparation” using pipelines.
def flatten_features(X):
    X = X.reshape(X.shape[0], -1)
    return X


cross_val_scores_28x28 = []
cross_val_scores_8x8 = []

print(mnist_28x28_train.shape)
flat_mnist_28x28_train = flatten_features(mnist_28x28_train)
print(flat_mnist_28x28_train.shape)

print(mnist_8x8_train.shape)
flat_mnist_8x8_train = flatten_features(mnist_8x8_train)
print(flat_mnist_8x8_train.shape)

for name, model in models.items():
    # define the pipeline

    steps = list()
    steps.append(('scaler', Normalizer()))
    steps.append((name, model))
    pipeline = Pipeline(steps=steps)

    n_splits = 5

    # define the evaluation procedure
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    # evaluate the model using cross-validation
    score = cross_val_score(pipeline, flat_mnist_28x28_train, train_labels, scoring='accuracy', cv=cv,
                            n_jobs=-1)
    cross_val_scores_28x28 = np.append(cross_val_scores_28x28, np.mean(score))

for name, model in models.items():
    # define the pipeline

    steps = list()
    steps.append(('scaler', Normalizer()))
    steps.append((name, model))
    pipeline = Pipeline(steps=steps)

    n_splits = 5

    # define the evaluation procedure
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    # evaluate the model using cross-validation
    score = cross_val_score(pipeline, flat_mnist_8x8_train, train_labels, scoring='accuracy', cv=cv,
                            n_jobs=-1)
    cross_val_scores_8x8 = np.append(cross_val_scores_8x8, np.mean(score))

# Cross validation plot
fig, ax = plt.subplots()

bar_width = 0.35
X = np.arange(6)

p1 = plt.bar(X, cross_val_scores_28x28, bar_width, color='r',
             label='28x28 dataset')

# The bar of second plot starts where the first bar ends
p2 = plt.bar(X + bar_width, cross_val_scores_8x8, bar_width,
             color='m',
             label='8x8 dataset')

plt.xlabel('Algorithms')
plt.ylabel('Cross Val Scores')
plt.title('Cross Val Performance of algorithms on the 28x28 vs 8x8 datasets')
plt.xticks(X + (bar_width / 2), (
    'GaussianNB', 'Dummy', 'DecisionTree', 'KNN', 'SVM', 'LogisticR'))
plt.legend()


def autolabel(ps):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in ps:
        height = np.round(rect.get_height(), 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(p1)
autolabel(p2)

plt.tight_layout()
plt.show()
