from exercise0_1 import iris
from exercise2_1 import X_test, Y_test
from exercise3_1 import feature_idx, mean_setosa, mean_versicolor, mean_virginica, sd_virginica, sd_versicolor, \
    sd_setosa
from exercise4_1 import normal_PDF
from exercise5_1_new import means, sds, priors, posterior
from exercise6_1 import classify
import numpy as np
from matplotlib import pyplot as plt


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


def decision_boundary(means, sds, priors):
    xs = np.linspace(1, 7, 1000)
    decision_boundaries = []
    # START ANSWER
    for x in xs:
        posteriors = []
        for i in range(3):
            posteriors.append(posterior(x, means, sds, priors, i))
        if (np.isclose(posteriors[0], posteriors[1], atol=1.e-1)):
            if 2 < x < 3:
                decision_boundaries.append(x)
        elif (np.isclose(posteriors[0], posteriors[2], atol=0)):
            decision_boundaries.append(x)
        elif (np.isclose(posteriors[1], posteriors[2], atol=1.e-2)):
            if 4 < x < 6:
                decision_boundaries.append(x)
        else:
            continue

    print(decision_boundaries)
    decision_boundaries

    plt.plot(xs, posterior(xs, means, sds, priors, 0), label=iris.target_names[0])
    plt.plot(xs, posterior(xs, means, sds, priors, 1), label=iris.target_names[1])
    plt.plot(xs, posterior(xs, means, sds, priors, 2), label=iris.target_names[2])

    # END ANSWER
    return decision_boundaries


# Create a scatterplot of the third and fourth feature.
feature_idx2 = 3
feature_idx = 2

plt.scatter(iris.data[:, feature_idx], iris.data[:, feature_idx2], c=iris.target)
plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel(iris.feature_names[feature_idx2])
decision_boundaries = decision_boundary(means, sds, priors)
for boundary in decision_boundaries:
    plt.axvline(x=boundary)

plt.show()
