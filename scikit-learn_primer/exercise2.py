from exercise1 import *
from sklearn.utils.validation import check_is_fitted

for name, model in models.items():
    # START ANSWER
    model.fit(X, y)
    # END ANSWER

for model in models.values():
    check_is_fitted(model)

from sklearn.metrics import f1_score, accuracy_score

for name, model in models.items():
    prediction = model.predict(X)
    f1_score_value = f1_score(prediction, y, average="weighted")
    accuracy = accuracy_score(prediction, y)
    # print(name)
    # print("- accuracy_score", accuracy)
    # print("- f1_score", f1_score_value)