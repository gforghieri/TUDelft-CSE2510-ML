from load_data import *

X_train.isnull().sum()
print(len(X_train))

indices_of_nan = np.array(np.where(X_train.isna())[0])

indices_of_nan = np.unique(indices_of_nan, axis=0)

print(len(y_train))
y_train = y_train.drop(index=indices_of_nan)
print(len(y_train))
X_train = X_train.dropna()

X_train.isnull().sum()
print(len(X_train))

X_test.isnull().sum()


# y_train = y_train.to_numpy()