from expe_2 import *

# use flat features
# pipes
# scaling
# gridsearch
# 1 plot with f1 score 1 plot with accuracy score mean

from sklearn.model_selection import GridSearchCV

random_state = 42
n_splits = 5
scoring_method = make_scorer(lambda prediction, true_target: f1_score(prediction, true_target, average="weighted"))

model_parameters = {
    "GaussianNB": {

    },
    "DummyClassifier": {

    },
    "DecisionTreeClassifier": {
        'DecisionTreeClassifier__random_state': [random_state],
        'DecisionTreeClassifier__max_depth': np.arange(1, 10),
        'DecisionTreeClassifier__min_samples_leaf': np.arange(1, 5)

    },
    # START ANSWER
    "KNeighborsClassifier": {
        'KNeighborsClassifier__n_neighbors': range(1, 5),
        'KNeighborsClassifier__weights': ['uniform', 'distance']
    },
    "SVM": {
        'SVM__random_state': [random_state],
        'SVM__C': np.arange(1, 15, 2),
        'SVM__kernel': ['linear', 'poly', 'rbf'],
    },
    "LogisticRegression": {
        'LogisticRegression__random_state': [random_state],
        'LogisticRegression__C': np.arange(1, 15, 2),
        'LogisticRegression__penalty': ['l1', 'l2', 'elasticnet', 'none']
    }
    # END ANSWER
}

gridcv_best_28x28 = []
gridcv_best_8x8 = []

for model_name, parameters in model_parameters.items():
    model = models[model_name]

    steps = list()
    steps.append(('scaler', Normalizer()))
    steps.append((model_name, model))
    pipeline = Pipeline(steps=steps)

    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1, verbose=False, scoring='accuracy').fit(
        mnist_28x28_train, train_labels)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    gridcv_best_28x28 = np.append(gridcv_best_28x28, best_score)

    print(model_name)
    print("- best_score =", best_score)
    print("best parameters:")
    for k, v in best_params.items():
        print("-", k, v)

for model_name, parameters in model_parameters.items():
    model = models[model_name]

    steps = list()
    steps.append(('scaler', Normalizer()))
    steps.append((model_name, model))
    pipeline = Pipeline(steps=steps)

    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1, verbose=False, scoring='accuracy').fit(
        mnist_8x8_train, train_labels)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    gridcv_best_8x8 = np.append(gridcv_best_8x8, best_score)

    print(model_name)
    print("- best_score =", best_score)
    print("best parameters:")
    for k, v in best_params.items():
        print("-", k, v)

# Cross validation plot
fig, ax = plt.subplots()

bar_width = 0.35
X = np.arange(6)

p1 = plt.bar(X, gridcv_best_28x28, bar_width, color='c',
             label='28x28 dataset')

# The bar of second plot starts where the first bar ends
p2 = plt.bar(X + bar_width, gridcv_best_8x8, bar_width,
             color='y',
             label='8x8 dataset')

plt.xlabel('Algorithms')
plt.ylabel('Tuned Accuracy Scores')
plt.title('Tuned GridSearchCV Performance of algorithms on the 28x28 vs 8x8 datasets')
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
