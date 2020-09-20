import numpy as np
from matplotlib import pyplot as plt
from exercise0_1 import iris
from exercise5_1_new import posterior, means, sds, priors

feature_idx = 2
xs = np.linspace(0, 7, 1000)
# START ANSWER
plt.plot(xs, posterior(xs, means, sds, priors, 0), label=iris.target_names[0])
plt.plot(xs, posterior(xs, means, sds, priors, 1), label=iris.target_names[1])
plt.plot(xs, posterior(xs, means, sds, priors, 2), label=iris.target_names[2])
# END ANSWER
plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel('Posterior probability')
plt.legend()
plt.show()
