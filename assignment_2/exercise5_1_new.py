import numpy as np

from exercise0_1 import iris
from exercise2_1 import setosa_X_train, X_train, versicolor_X_train, Y_train, virginica_X_train
from exercise3_1 import mean_setosa, mean_versicolor, mean_virginica, sd_virginica, sd_setosa, sd_versicolor, \
    feature_idx
from exercise4_1 import normal_PDF


def posterior(x, means, sds, priors, i):
    """
    Compute the posterior probability P(C_i | x).
    :param x: the sample to compute the posterior probability for.
    :param means: an array of means for each class.
    :param sds: an array of standard deviation values for each class.
    :param priors: an array of frequencies for each class.
    :param i: the index of the class to compute the posterior probability for.
    """
    posterior = 0
    # START ANSWER
    numerator = normal_PDF(x, means[i], sds[i]) * priors[i]
    all_pdfs = []
    for k in range(3):
        all_pdfs = np.append(all_pdfs, normal_PDF(x, means[k], sds[k]))
    denominator = np.sum(all_pdfs * priors)
    posterior = numerator / denominator
    # END ANSWER
    return posterior


means = [mean_setosa, mean_versicolor, mean_virginica]
sds = [sd_setosa, sd_versicolor, sd_virginica]
priors = [
    setosa_X_train.shape[0] / X_train.shape[0],
    versicolor_X_train.shape[0] / X_train.shape[0],
    virginica_X_train.shape[0] / X_train.shape[0]
]

# Test out the code
flower_idx = 6
print("Flower belongs to class", iris.target_names[Y_train[flower_idx]])

# iterate over all classes
for i in range(3):
    x_post = posterior(X_train[flower_idx, feature_idx], means, sds, priors, i)
    print("Posterior probability for class", iris.target_names[i], ": ", x_post)

post_setosa = posterior(X_train[flower_idx, feature_idx], means, sds, priors, 0)
post_versicolor = posterior(X_train[flower_idx, feature_idx], means, sds, priors, 1)
post_virginica = posterior(X_train[flower_idx, feature_idx], means, sds, priors, 2)

assert np.isclose(post_setosa, 1.1048294835009998e-107, rtol=0.0001,
                  atol=0.), "Expected a different posterior probability"
assert np.isclose(post_versicolor, 0.03817178391547811, rtol=0.0001,
                  atol=0.), "Expected a different posterior probability"
assert np.isclose(post_virginica, 0.9618282160845218, rtol=0.0001,
                  atol=0.), "Expected a different posterior probability"
