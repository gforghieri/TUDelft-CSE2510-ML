from load_data import *
from collections import Counter

print(X_train.head())

print(X_train.describe)

X_train.isnull().sum()

print(np.where(X_train['occupation'].isnull())[0])
print(np.where(X_train['workclass'].isna())[0])

for i in range(len(X_train.columns)):
    print(X_train[X_train.columns[i]].value_counts())
