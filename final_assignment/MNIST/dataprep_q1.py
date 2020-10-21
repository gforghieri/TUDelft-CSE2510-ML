from sklearn.preprocessing import MinMaxScaler

from dataexpl_q1 import *

# flatten the features of the data, instead of (3750,8,8 to 3750,64)
flat_mnist_8x8_train = mnist_8x8_train.reshape(mnist_8x8_train.shape[0], -1)
flat_mnist_8x8_test = mnist_8x8_test.reshape(mnist_8x8_test.shape[0], -1)

# flatten the features of the data, instead of (3750,28,28) to (3750,784)
flat_mnist_28x28_train = mnist_28x28_train.reshape(mnist_28x28_train.shape[0], -1)
flat_mnist_28x28_test = mnist_28x28_train.reshape(mnist_28x28_test.shape[0], -1)

# scale both train and test features from rgb (0-255) to grayscale (0-1)
min_max_scaler = MinMaxScaler()

scaled_mnist_8x8_train = min_max_scaler.fit_transform(X=flat_mnist_8x8_train)
scaled_mnist_8x8_test = min_max_scaler.fit_transform(X=flat_mnist_8x8_test)

scaled_mnist_28x28_train = min_max_scaler.fit_transform(X=flat_mnist_28x28_train)
scaled_mnist_28x28_test = min_max_scaler.fit_transform(X=flat_mnist_28x28_test)