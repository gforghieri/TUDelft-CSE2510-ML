from exercise2 import *

from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True, stratify=y)

# START ANSWER
for name, model in models.items():
    # START ANSWER
    model.fit(X_train, y_train)
    # END ANSWER

for name, model in models.items():
    prediction = model.predict(X_test)
    f1_score_value = f1_score(prediction, y_test, average="weighted")
    accuracy = accuracy_score(prediction, y_test)
    print(name)
    print("- accuracy_score", accuracy)
    print("- f1_score", f1_score_value)

# END ANSWER

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


def k_fold_fit_and_evaluate(X, y, model, scoring_method, n_splits=5):
    # define evaluation procedure
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    # evaluate model
    scores = cross_validate(model, X, y, scoring=scoring_method, cv=cv, n_jobs=-1)

    return scores["test_score"]

