from exercise3 import *

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)

scaler.mean_, scaler.scale_
print(scaler.mean_, scaler.scale_)

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
# First, transform the dataset using the Normalizer transformer.
# Then fit and evaluate each model using the transformed features.

normalized_X_train = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)

for name, model in models.items():
    model.fit(normalized_X_train, y_train)

for name, model in models.items():
    prediction = model.predict(normalized_X_test)
    f1_score_value = f1_score(prediction, y_test, average="weighted")
    accuracy = accuracy_score(prediction, y_test)
    print(name)
    print("- accuracy_score", accuracy)
    print("- f1_score", f1_score_value)

# END ANSWER
