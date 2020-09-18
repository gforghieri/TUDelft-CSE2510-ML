import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split  # to split in train and test set

# load the data and create the training and test sets
iris = datasets.load_iris()

# START ANSWER
setosa_flowers = iris.data[np.where(iris.target == 0)]
versicolor_flowers = iris.data[np.where(iris.target == 1)]
virginica_flowers = iris.data[np.where(iris.target == 2)]
# END ANSWER

# X is the feature vectors for the data points, and Y is the target (ground truth) class for those data points
# the iris.data and iris.target entries are randomly divided into training and test sets.
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=20)

# Due to the randomness of the split, number of each flowers is not necessarily the same
# Separate the training dataset into the three flower types.
setosa_X_train = None
versicolor_X_train = None
virginica_X_train = None

# START ANSWER
setosa_X_train = X_train[np.where(Y_train == 0)]
versicolor_X_train = X_train[np.where(Y_train == 1)]
virginica_X_train = X_train[np.where(Y_train == 2)]
# END ANSWER

assert setosa_X_train.shape[0] != versicolor_X_train.shape[0]
assert setosa_X_train.shape[0] != virginica_X_train.shape[0]
assert versicolor_X_train.shape[0] != virginica_X_train.shape[0]

setosa_X_train.shape, versicolor_X_train.shape, virginica_X_train.shape

# We use the third feature
feature_idx = 2
iris.feature_names

# From the Matplotlib library, import pyplot. We will refer to this library later as plt.
# This is a widely used library that lets you create images and plot your data.
from matplotlib import pyplot as plt

plt.hist(setosa_flowers[:, feature_idx], label=iris.target_names[0])
plt.hist(versicolor_flowers[:, feature_idx], label=iris.target_names[1])
plt.hist(virginica_flowers[:, feature_idx], label=iris.target_names[2])
plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel('Number of flowers')
plt.legend()
plt.show()


def compute_mean(x):
    mean = 0
    # START ANSWER
    mean = np.sum(x) / len(x)
    # END ANSWER
    return mean


def compute_sd(x, mean):
    sd = 0
    # START ANSWER


    # END ANSWER
    return sd


# Compute the mean for each flower type.
mean_setosa = compute_mean(setosa_X_train[:, feature_idx])
mean_versicolor = compute_mean(versicolor_X_train[:, feature_idx])
mean_virginica = compute_mean(virginica_X_train[:, feature_idx])

# Compute the standard deviation for each flower type.
sd_setosa = compute_sd(setosa_X_train[:, feature_idx], mean_setosa)
sd_versicolor = compute_sd(versicolor_X_train[:, feature_idx], mean_versicolor)
sd_virginica = compute_sd(virginica_X_train[:, feature_idx], mean_virginica)

# Print the computed means and standard deviations.
print("setosa", mean_setosa, sd_setosa)
print("versicolor", mean_versicolor, sd_versicolor)
print("virginica", mean_virginica, sd_virginica)

assert np.isclose(mean_setosa, 1.4729729729729728), "Expected a different mean"
assert np.isclose(mean_versicolor, 4.25), "Expected a different mean"
assert np.isclose(mean_virginica, 5.572222222222222), "Expected a different mean"

assert np.isclose(sd_setosa, 0.17652600857089654), "Expected a different standard deviation"
assert np.isclose(sd_versicolor, 0.44300112866673375), "Expected a different standard deviation"
assert np.isclose(sd_virginica, 0.547017728288333), "Expected a different standard deviation"
