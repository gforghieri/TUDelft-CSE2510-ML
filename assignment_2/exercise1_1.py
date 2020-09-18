import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris

print("First five flowers: \n", iris.data[:5, :])
print("Their labels: ", iris.target[:5])
print("And the label names: ", iris.target_names)

last_five_flowers = iris.data[len(iris.data) - 5:, :]
third_feature_only = iris.data[:, 2]
first_ten_names = None

# START ANSWER
first_ten_target = iris.target[0:10]
first_ten_names = np.arange(10).astype(str)
first_ten_names = np.where(first_ten_target == 0, iris.target_names[0],
                           np.where(first_ten_target == 1, iris.target_names[1],
                                    np.where(first_ten_target == 2, iris.target_names[2], 'wrong_type')))
# # END ANSWER

setosa_flowers = None
versicolor_flowers = None
virginica_flowers = None
# START ANSWER
setosa_flowers = iris.data[np.where(iris.target == 0)]
versicolor_flowers = iris.data[np.where(iris.target == 1)]
virginica_flowers = iris.data[np.where(iris.target == 2)]
# END ANSWER


print("Last five flowers: \n", last_five_flowers)
print("Only the third feature: ", third_feature_only)
print("All label names: ", first_ten_names)

print("Class: ", iris.target_names[0], "; Items: \n", setosa_flowers)

assert last_five_flowers.shape == (5, 4), "Expected a two dimensional array of shape (5,4)"
assert third_feature_only.shape == (150,), "Expected an array of shape (150,)"
assert first_ten_names.shape == (10,), "Expected an array of shape (10,)"

assert setosa_flowers.shape == (50, 4), "Expected a two dimensional array of shape (50,4)"
assert versicolor_flowers.shape == (50, 4), "Expected a two dimensional array of shape (50,4)"
assert virginica_flowers.shape == (50, 4), "Expected a two dimensional array of shape (50,4)"
