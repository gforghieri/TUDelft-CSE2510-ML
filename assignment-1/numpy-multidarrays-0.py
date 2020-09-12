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