from load_data import *

X_train.isnull().sum()
len(X_train)

X_train = X_train.dropna()

X_train.isnull().sum()
len(X_train)