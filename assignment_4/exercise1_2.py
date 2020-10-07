import scipy
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the digits with 2 classes (0 and 1)

binary_digits = datasets.load_digits(n_class=2)
binary_digits_images = binary_digits.images
binary_digits_labels = binary_digits.target

# for i in range(10):
#     digit_image = binary_digits_images[i,:,:]
#     plt.figure()
#     plt.gray()
#     plt.title("digit: " + str(binary_digits_labels[i]))
#     plt.imshow(digit_image)
#     # plt.show()