import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt


def normal_PDF(x, mean, sd):
    pdf = 0
    # START ANSWER
    pdf = 1 / (np.sqrt(2 * np.pi * (sd ** 2)))
    e_expression = np.e ** -(((x - mean) ** 2) / (2 * (sd ** 2)))
    pdf = pdf * e_expression
    # END ANSWER
    return pdf


# Set x, mean and standard deviation
x = 0.5
mean = 5
sd = 0.5
my_pdf = normal_PDF(x, mean, sd)

# You can compare your outcome to scipy's built-in normal PDF
scipy_pdf = norm.pdf(x, mean, sd)
print("Your pdf function outcome: ", my_pdf, " Scipy's function outcome: ", scipy_pdf)
assert np.isclose(my_pdf, scipy_pdf)

# And we plot the result of your PDF function for 100 points between 0 and 4: np.linspace(0, 4, 100)
xs = np.linspace(0, 10, 100)
plt.plot(xs, normal_PDF(xs, mean, sd))
plt.show()
