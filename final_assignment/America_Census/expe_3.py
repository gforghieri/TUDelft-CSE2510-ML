from numpy import mean, std
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.preprocessing import DenseTransformer

from expe_1 import *

grid_accuracy_scores = []
grid_f1_scores = []
grid_roc_auc_scores = []

model_names = []

scoring_method_f1 = make_scorer(lambda prediction, true_target: f1_score(prediction, true_target, average="weighted"))
scoring_method_accuracy = make_scorer(accuracy_score)
scoring_method_roc_auc = make_scorer(roc_auc_score)


# evaluate a model
def evaluate_model_gridsearch(X, y, model, scorer, parameters):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)
    # evaluate model
    grid_search = GridSearchCV(model, parameters, cv=cv, n_jobs=6, verbose=False, scoring=scorer).fit(X, y)
    return grid_search


model_parameters = {
    "GaussianNB": {

    },
    "DummyClassifier": {

    },
    "DecisionTreeClassifier": {
        'DecisionTreeClassifier__random_state': [random_state],
        'DecisionTreeClassifier__max_depth': np.arange(1, 15),
        'DecisionTreeClassifier__min_samples_leaf': np.arange(1, 10)

    },
    # START ANSWER
    "KNeighborsClassifier": {
        'KNeighborsClassifier__n_neighbors': range(1, 10),
        'KNeighborsClassifier__weights': ['uniform', 'distance']
    },
    "SVM": {
        'SVM__random_state': [random_state],
        'SVM__C': np.arange(0.1, 15, 2),
        'SVM__kernel': ['linear', 'poly', 'rbf'],
    },
    "LogisticRegression": {
        'LogisticRegression__random_state': [random_state],
        'LogisticRegression__C': np.arange(0.1, 2, 15),
        'LogisticRegression__penalty': ['l1', 'l2', 'elasticnet', 'none']
    }
    # END ANSWER
}

# evaluate each model
for model_name, parameters in model_parameters.items():
    model = models[model_name]
    # define steps
    steps = list()
    steps.append(('c', OneHotEncoder(handle_unknown='ignore'), cat_columns_train))
    steps.append(('n', MinMaxScaler(), num_columns_train))
    # one hot encode categorical, normalize numerical
    ct = ColumnTransformer(steps)
    # wrap the model i a pipeline
    pipeline = Pipeline(steps=[('t', ct), ('to_dense', DenseTransformer()), (model_name, model)])
    # evaluate the model and store results
    grid_search_acc = evaluate_model_gridsearch(X_train, y_train.values.ravel(), pipeline,
                                                scorer=scoring_method_accuracy, parameters=parameters)
    acc_best_model = grid_search_acc.best_estimator_
    acc_best_score = grid_search_acc.best_score_
    acc_best_params = grid_search_acc.best_params_
    grid_accuracy_scores.append(acc_best_score)
    print(model_name)
    print("- acc_best_score =", acc_best_score)
    print("acc_best parameters:")
    for k, v in acc_best_params.items():
        print("-", k, v)

    grid_search_f1 = evaluate_model_gridsearch(X_train, y_train.values.ravel(), pipeline, scorer=scoring_method_f1,
                                               parameters=parameters)
    f1_best_model = grid_search_f1.best_estimator_
    f1_best_score = grid_search_f1.best_score_
    f1_best_params = grid_search_f1.best_params_
    grid_f1_scores.append(f1_best_score)
    print(model_name)
    print("- f1_best_score =", f1_best_score)
    print("f1_best parameters:")
    for k, v in f1_best_params.items():
        print("-", k, v)

    grid_search_roc = evaluate_model_gridsearch(X_train, y_train.values.ravel(), pipeline,
                                                scorer=scoring_method_roc_auc, parameters=parameters)
    roc_best_model = grid_search_roc.best_estimator_
    roc_best_score = grid_search_roc.best_score_
    roc_best_params = grid_search_roc.best_params_
    grid_roc_auc_scores.append(roc_best_score)
    print(model_name)
    print("- roc_best_score =", roc_best_score)
    print("roc_best parameters:")
    for k, v in roc_best_params.items():
        print("-", k, v)

    # f1 = evaluate_model_gridsearch(X_train, y_train.values.ravel(), pipeline, scorer=scoring_method_f1,
    #                                parameters=parameters)
    # f1_scores.append(np.mean(f1))
    # auc_sco = evaluate_model_gridsearch(X_train, y_train.values.ravel(), pipeline, scorer=scoring_method_roc_auc,
    #                                     parameters=parameters)
    # roc_auc_scores.append(np.mean(auc_sco))
    # model_names.append(model_name)
    # # summarize performance
    # print("acc score")
    # print('>%s %.3f (%.3f)' % (name, mean(acc_score), std(acc_score)))
    # print("f1 score")
    # print('>%s %.3f (%.3f)' % (name, mean(f1), std(f1)))
    # print("auc-roc score")
    # print('>%s %.3f (%.3f)' % (name, mean(auc_sco), std(auc_sco)))
# plot the results
# plt.boxplot(accuracy_scores, labels=model_names, showmeans=True)
# plt.show()

# Cross validation plot
fig, ax = plt.subplots()

bar_width = 0.25
X = np.arange(6)

p1 = plt.bar(X + 0.00, grid_accuracy_scores, bar_width, color='c',
             label='accuracy')

# The bar of second plot starts where the first bar ends
p2 = plt.bar(X + 0.25, grid_f1_scores, bar_width,
             color='m',
             label='f1 score')

# The bar of second plot starts where the first bar ends
p3 = plt.bar(X + 0.50, grid_roc_auc_scores, bar_width,
             color='y',
             label='roc-auc score')

plt.xlabel('Algorithms')
plt.ylabel('Tuned Performance scores')
plt.title('Tuned classification metrics of algorithms on US Census data')
plt.xticks(X + (bar_width + bar_width / 2), (
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
autolabel(p3)

plt.tight_layout()
plt.show()
