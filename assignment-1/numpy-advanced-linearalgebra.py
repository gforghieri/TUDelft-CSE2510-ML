# Calculate the inner product of two vector v and w both of shape (N,).
# Validate your result by computing the dot product using multiply and sum operations.
import numpy as np

v = np.arange(5)
w = np.arange(5, 10)

# START ANSWER
dot_product = np.dot(v, w)

# END ANSWER

assert (dot_product == np.sum(v * w))
