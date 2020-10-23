from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer

from dataexpl_q1 import *

# 1. split data
# 2. flatten 3d to 2d
# 3. scale
# 4. fit
# 5. predict
# 6. evaluate
# 7. graph/plot
# 8. analyze


# 1. split data
X_train_28x28, X_test_28x28, y_train, y_test = train_test_split(mnist_28x28_train, train_labels, test_size=0.3,
                                                                random_state=42, shuffle=True, stratify=train_labels)
X_train_8x8, X_test_8x8, y_train, y_test = train_test_split(mnist_8x8_train, train_labels, test_size=0.3,
                                                            random_state=42, shuffle=True, stratify=train_labels)

# 2. flatten 3d to 2d
# flatten the features of the data, instead of (,28,28) to (,784)
flat_X_train_28x28 = X_train_28x28.reshape(X_train_28x28.shape[0], -1)
flat_X_test_28x28 = X_test_28x28.reshape(X_test_28x28.shape[0], -1)

# flatten the features of the data, instead of (,8,8 to ,64)
flat_X_train_8x8 = X_train_8x8.reshape(X_train_8x8.shape[0], -1)
flat_X_test_8x8 = X_test_8x8.reshape(X_test_8x8.shape[0], -1)

# 3. scale
# scale both train and test features from rgb (0-255) to black & whtie (0-1)
min_max_scaler_28x28 = MinMaxScaler()
min_max_scaler_8x8 = MinMaxScaler()

min_max_scaler_28x28.fit(flat_X_train_28x28)
min_max_scaler_8x8.fit(flat_X_train_8x8)

scaled_X_train_28x28 = min_max_scaler_28x28.transform(flat_X_train_28x28)
scaled_X_test_28x28 = min_max_scaler_28x28.transform(flat_X_test_28x28)

scaled_X_train_8x8 = min_max_scaler_8x8.transform(X=flat_X_train_8x8)
scaled_X_test_8x8 = min_max_scaler_8x8.transform(X=flat_X_test_8x8)

# scaled_X_train_28x28 = flat_X_train_28x28
# scaled_X_test_28x28 = flat_X_test_28x28
#
# scaled_X_train_8x8 = flat_X_train_8x8
# scaled_X_test_8x8 = flat_X_test_8x8
