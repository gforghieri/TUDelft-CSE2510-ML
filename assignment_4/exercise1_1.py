import scipy
import sklearn
import numpy as np
import matplotlib.pyplot as plt


from sklearn import datasets

# Load the digits with 10 classes (digits 0 - 9)
all_digits = datasets.load_digits(n_class=10)
all_digits_images = all_digits.images
all_digits_labels = all_digits.target

'''
all_digits_images is a numpy array where:
- the first index is the index of individual images
- the second index corresponds to the row of the pixel
- the third index corresponds to the column of the pixel
i.e.: all_digits_images[image_index,row,column]
the values of the pixels are values between 0 (black) and 16 (white)
'''

for i in range(10):
    digit_image = all_digits_images[i]
    plt.figure()
    plt.gray()
    plt.title("digit: " + str(all_digits_labels[i]))
    plt.imshow(digit_image)
    plt.show()