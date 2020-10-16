from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)

scaler.mean_, scaler.scale_

X_train_transformed = scaler.transform(X_train)
print("X_train")
print("mean", X_train.mean())
print("std", X_train.std())
print()
print("X_train_transformed")
print("mean", X_train_transformed.mean())
print("std", X_train_transformed.std())


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True, stratify=y)

scaler = preprocessing.Normalizer()

# START ANSWER
# END ANSWER