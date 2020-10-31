from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from dataprep_2 import *

# split data
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2,
                                                                            random_state=42, shuffle=True,
                                                                            stratify=y_train)

steps = list()
steps.append(('c', OneHotEncoder(handle_unknown='ignore'), cat_columns_train))
steps.append(('n', MinMaxScaler(), num_columns_train))

# define steps
steps = [('c', OneHotEncoder(handle_unknown='ignore'), cat_columns_train), ('n', MinMaxScaler(), num_columns_train)]
# one hot encode categorical, normalize numerical
ct = ColumnTransformer(steps)
