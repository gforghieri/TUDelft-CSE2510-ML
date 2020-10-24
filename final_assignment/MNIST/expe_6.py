from sklearn.preprocessing import Normalizer

from load_data_3 import *

def flatten_features(X):
    X = X.reshape(X.shape[0], -1)
    return X

final_X_train_28x28 = mnist_28x28_train.copy()
flat_final_X_train_28x28 = flatten_features(final_X_train_28x28)

final_X_test_28x28 = mnist_28x28_test.copy()
flat_final_X_test_28x28 = flatten_features(final_X_test_28x28)


scaler = Normalizer()

scaler.fit(flat_final_X_train_28x28)

scaled_final_X_train_28x28 = scaler.transform(flat_final_X_train_28x28)
scaled_final_X_test_28x28 = scaler.transform(flat_final_X_test_28x28)

final_SVC_clf = SVC(kernel='poly', C=3, random_state=42)

final_SVC_clf.fit(scaled_final_X_train_28x28, train_labels)

final_prediction = final_SVC_clf.predict(scaled_final_X_test_28x28)

print(final_prediction)

prediction = np.array([-1] * len(final_prediction)) #TODO replace this with you own prediction
pd.DataFrame(prediction).to_csv("GROUP_classes_problem_mnist.txt", index=False, header=False)