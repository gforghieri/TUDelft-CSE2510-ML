from exercise4 import *

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
        'random_state': [random_state],
        'max_depth': [None, 2, 5, 10]
    },
    # START ANSWER
    "KNeighborsClassifier": {
        'n_neighbors': range(1, 3),
        'weights': ['uniform', 'distance']
    },
    "SVM": {
        'C': np.arange(0.5, 15, 0.5)
    },
    "LogisticRegression": {
        'C': np.arange(0.5, 1, 0.5)
    }
    # END ANSWER
}

for model_name, parameters in model_parameters.items():
    model = models[model_name]

    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    grid_search = GridSearchCV(model, parameters, cv=cv, n_jobs=-1, verbose=False, scoring=scoring_method).fit(X, y)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    # print(model_name)
    # print("- best_score =", best_score)
    # print("best paramters:")
    # for k, v in best_params.items():
    #     print("-", k, v)
