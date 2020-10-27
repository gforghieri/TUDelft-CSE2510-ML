from sklearn.model_selection import train_test_split

from dataprep_3 import *

# split data
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2,
                                                                random_state=42, shuffle=True, stratify=y_train)

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
