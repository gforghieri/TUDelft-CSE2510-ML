import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split  # to split in train and test set

# load the data and create the training and test sets
iris = datasets.load_iris()
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
