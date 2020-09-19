import numpy as np

from exercise0_1 import iris
from exercise2_1 import X_train, Y_train
from exercise3_1 import feature_idx
from exercise5_1_new import means, sds, priors, posterior


def classify(x, means, sds, priors):
    classification = -1
    # START ANSWER
    setosa_probability = posterior(x, means, sds, priors, 0)
    versicolor_probability = posterior(x, means, sds, priors, 1)
    virginica_probability = posterior(x, means, sds, priors, 2)
    classification = np.argmax([setosa_probability, versicolor_probability, virginica_probability])
    # END ANSWER
    return classification

# Test out the code
flower_idxs = [5,20,30]
predicted_classes = np.zeros(3, dtype=np.int64)
for i, flower_idx in enumerate(flower_idxs):
    predicted_classes[i] = classify(X_train[flower_idx, feature_idx], means, sds, priors)

print("Predicted class", iris.target_names[predicted_classes])
print("Flower belongs to class", iris.target_names[Y_train[flower_idxs]])
assert (predicted_classes == Y_train[flower_idxs]).all()