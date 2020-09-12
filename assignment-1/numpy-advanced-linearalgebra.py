# Calculate the inner product of two vector v and w both of shape (N,).
# Validate your result by computing the dot product using multiply and sum operations.
import numpy as np

v = np.arange(5)
w = np.arange(5, 10)

# START ANSWER
dot_product = np.dot(v, w)

# END ANSWER

assert (dot_product == np.sum(v * w))

# Calculate the product of a matrix A of shape (M,N) with a vector v of shape (N,).
m = n = 5
X = np.arange(m * n).reshape(m, n)
v = np.arange(n).reshape(n)

product = None
# START ANSWER
product = np.matmul(X, v)
# END ANSWER

assert (product == np.array([30, 80, 130, 180, 230])).all()
product
