import numpy as np
from sklearn import datasets # to load the dataset
from sklearn.model_selection import train_test_split #to split in train and test set

seed = 20
# load the data and create the training and test sets
iris = datasets.load_iris()
# X is the feature vectors of the data points, and Y is the target (ground truth) class for those data points
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=seed)

from matplotlib import pyplot as plt

# # START ANSWER
# # Create a scatterplot of the third and fourth feature of the training data set.
# plt.scatter(X_train[:, 2], X_train[:, 3], c=Y_train)
# plt.xlabel(iris.feature_names[2])
# plt.ylabel(iris.feature_names[3])
# plt.show()
# # END ANSWER