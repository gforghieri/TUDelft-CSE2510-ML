import numpy as np

from exercise3 import *
from sklearn.metrics import make_scorer

n_splits = 5

prediction

scoring_method_f1 = make_scorer(lambda prediction, true_target: f1_score(prediction, true_target, average="weighted"))
# START ANSWER
scoring_method_accuracy = make_scorer(accuracy_score)

# END ANSWER


for name, model in models.items():
    print(name)
    metrics_f1 = k_fold_fit_and_evaluate(X, y, model, scoring_method_f1, n_splits=n_splits)
    # START ANSWER
    mean_f1 = np.mean(metrics_f1)
    print(mean_f1)
    std_f1 = np.std(metrics_f1)
    print(std_f1)

    metrics_accuracy = k_fold_fit_and_evaluate(X, y, model, scoring_method_accuracy, n_splits=n_splits)

    mean_accuracy = np.mean(metrics_accuracy)
    std_accuracy = np.std(metrics_accuracy)

    print(mean_accuracy)
    print(std_accuracy)
    # END ANSWER