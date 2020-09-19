from exercise2_1 import X_test, Y_test
from exercise3_1 import feature_idx
from exercise5_1_new import means, sds, priors
from exercise6_1 import classify
import numpy as np


def evaluate(X_test, Y_test, means, sds, priors):
    accuracy = 0

    predicted_classes = np.zeros(len(X_test))
    test_petal_lengths = X_test[:, feature_idx]

    # START ANSWER
    for i, test_petal_length in enumerate(test_petal_lengths):
        predicted_classes[i] = classify(test_petal_length, means, sds, priors)

    counter = 0
    for i in range(len(predicted_classes)):
        if predicted_classes[i] == Y_test[i]:
            counter += 1

    accuracy = counter / len(Y_test)

    # END ANSWER
    return accuracy


accuracy = evaluate(X_test, Y_test, means, sds, priors)

print(accuracy)
assert accuracy > 0.9, "Expected a higher accuracy"
