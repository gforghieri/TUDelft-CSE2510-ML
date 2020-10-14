from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

random_state = 42

models = {
    "GaussianNB": GaussianNB(),
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=random_state),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3, weights="distance"),
    # START ANSWER
    # END ANSWER
}

assert "GaussianNB" in models and isinstance(models["GaussianNB"], GaussianNB), "There is no GaussianNB in models"
assert "DecisionTreeClassifier" in models and isinstance(models["DecisionTreeClassifier"], DecisionTreeClassifier), "There is no DecisionTreeClassifier in models"
assert "KNeighborsClassifier" in models and isinstance(models["KNeighborsClassifier"], KNeighborsClassifier), "There is no KNeighborsClassifier in models"
assert "SVM" in models and isinstance(models["SVM"], SVC), "There is no SVC in models"
assert "LogisticRegression" in models and isinstance(models["LogisticRegression"], LogisticRegression), "There is no LogisticRegression in models"