from numpy import mean, std
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from mlxtend.preprocessing import DenseTransformer

from expe_1 import *

accuracy_scores = []
f1_scores = []
roc_auc_scores = []

model_names = []

scoring_method_f1 = make_scorer(lambda prediction, true_target: f1_score(prediction, true_target, average="weighted"))
scoring_method_accuracy = make_scorer(accuracy_score)
scoring_method_roc_auc = make_scorer(roc_auc_score)


# scoring_method_f1 = 'f1'
# # START ANSWER
# scoring_method_accuracy = 'accuracy'


# evaluate a model
def evaluate_model(X, y, model, scorer):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=6)
    return scores


# evaluate each model
for name, model in models.items():
    # define steps
    steps = list()
    steps.append(('c', OneHotEncoder(handle_unknown='ignore'), cat_columns_train))
    steps.append(('n', MinMaxScaler(), num_columns_train))
    # one hot encode categorical, normalize numerical
    ct = ColumnTransformer(steps)
    # wrap the model i a pipeline
    pipeline = Pipeline(steps=[('t', ct), ('to_dense', DenseTransformer()), ('m', model)])
    # evaluate the model and store results
    acc_score = evaluate_model(X_train, y_train.values.ravel(), pipeline, scorer=scoring_method_accuracy)
    accuracy_scores.append(np.mean(acc_score))
    f1 = evaluate_model(X_train, y_train.values.ravel(), pipeline, scorer=scoring_method_f1)
    f1_scores.append(np.mean(f1))
    auc_sco = evaluate_model(X_train, y_train.values.ravel(), pipeline, scorer=scoring_method_roc_auc)
    roc_auc_scores.append(np.mean(auc_sco))
    model_names.append(name)
    # summarize performance
    print("acc score")
    print('>%s %.3f (%.3f)' % (name, mean(acc_score), std(acc_score)))
    print("f1 score")
    print('>%s %.3f (%.3f)' % (name, mean(f1), std(f1)))
    print("auc-roc score")
    print('>%s %.3f (%.3f)' % (name, mean(auc_sco), std(auc_sco)))
# plot the results
# plt.boxplot(accuracy_scores, labels=model_names, showmeans=True)
# plt.show()

# Cross validation plot
fig, ax = plt.subplots()

bar_width = 0.25
X = np.arange(6)

p1 = plt.bar(X + 0.00, accuracy_scores, bar_width, color='g',
             label='accuracy')

# The bar of second plot starts where the first bar ends
p2 = plt.bar(X + 0.25, f1_scores, bar_width,
             color='b',
             label='f1 score')

# The bar of second plot starts where the first bar ends
p3 = plt.bar(X + 0.50, roc_auc_scores, bar_width,
             color='r',
             label='roc-auc score')

plt.xlabel('Algorithms')
plt.ylabel('Performance scores')
plt.title('Different classification metrics of algorithms on US Census data')
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
