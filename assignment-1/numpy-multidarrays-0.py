import numpy as np

m = n = 5
X = np.arange(m * n).reshape(m, n)

print(X)

# Select the j-th column from the matrix X. What happens if you use X[j]? Is this correct?
j = 3
column = None
# START ANSWER
column = X[:, j]
# END ANSWER

assert (column == np.array([ 3,  8, 13, 18, 23])).all()
column

# Given an ndarray X with shape (m,n), calculate the mean of each column.
# Try doing this without using np.mean or loops.
# Hint: try to sum up the entries and dividing them by the number of elements

means = 0
means = np.arange(m)
# START ANSWER
for i in means:
    means[i] = np.sum(X[:, i]) / len(X[:, i])

# END ANSWER

assert (means == np.array([10., 11., 12., 13., 14.])).all()
means