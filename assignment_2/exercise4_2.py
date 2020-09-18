import numpy as np
from matplotlib import pyplot as plt
from exercise0_1 import iris
from exercise2_1 import setosa_X_train, versicolor_X_train, virginica_X_train
from exercise3_1 import feature_idx, mean_setosa, sd_setosa, mean_versicolor, sd_versicolor, sd_virginica, \
    mean_virginica

# Histograms of the flower types of the training set
from exercise4_1 import normal_PDF

plt.hist(setosa_X_train[:, feature_idx], label=iris.target_names[0])
plt.hist(versicolor_X_train[:, feature_idx], label=iris.target_names[1])
plt.hist(virginica_X_train[:, feature_idx], label=iris.target_names[2])

# Plot your PDFs here
xs = np.linspace(0, 7, 100)

# START ANSWER
plt.plot(xs, normal_PDF(xs, mean_setosa, sd_setosa))
plt.plot(xs, normal_PDF(xs, mean_versicolor, sd_versicolor))
plt.plot(xs, normal_PDF(xs, mean_virginica, sd_virginica))

# END ANSWER

plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel('Number of flowers / PDF')
plt.legend()
plt.show()
