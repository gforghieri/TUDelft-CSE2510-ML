from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

from dataprep_1 import *

# select columns with numerical data types
num_columns_train = X_train.select_dtypes(include=['int64', 'float64']).columns
# select a subset of the dataframe with the numerical columns
num_subset_train = X_train[num_columns_train]
print(num_subset_train)

# select columns with numerical data types
num_columns_test = X_test.select_dtypes(include=['int64', 'float64']).columns
# select a subset of the dataframe with the numerical columns
num_subset_test = X_test[num_columns_test]
print(num_subset_train)

cat_columns_train = X_train.select_dtypes(include=['object']).columns
cat_subset_train = X_train[cat_columns_train]

cat_columns_test = X_test.select_dtypes(include=['object']).columns
cat_subset_test = X_test[cat_columns_test]



