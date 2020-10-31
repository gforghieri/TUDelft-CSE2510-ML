from mlxtend.preprocessing import DenseTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from dataprep_1 import *
from dataprep_2 import cat_columns_train, num_columns_train

steps = list()
steps.append(('c', OneHotEncoder(handle_unknown='ignore'), cat_columns_train))
steps.append(('n', MinMaxScaler(), num_columns_train))

ct = ColumnTransformer(steps)

# THIS IS TO DO
# WHat model?
# WHat parameters?
final_clf = LogisticRegression(C=0.1, penalty='none', random_state=42)  # TODO: Include tuned parameters

pipeline = Pipeline(steps=[('t', ct), ('to_dense', DenseTransformer()), ('insert-modelname', final_clf)])

pipeline.fit(X_train, y_train.values.ravel())

final_prediction = pipeline.predict(X_test)

prediction = np.array(final_prediction)  # TODO replace this with you own prediction
pd.DataFrame(prediction).to_csv("GROUP_classes_problem_census.txt", index=False, header=False)
