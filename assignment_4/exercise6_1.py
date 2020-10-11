from exercise5_3 import *

# Import the load function for the dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split

n_classes = 10

# Load the digits with 10 classes (digits 0 - 9)
all_digits = datasets.load_digits(n_class=n_classes)
all_digits_images = all_digits.images
all_digits_labels = all_digits.target

# flatten de data so they are 1D and append extra ones to the feature vectors
all_digits_pixels = add_one_features(all_digits_images.reshape(all_digits_images.shape[0], -1))

# the shape should be (1797, 65)
print(all_digits_pixels.shape)

# Split dataset into train and test set
x_train_digits, x_test_digits, y_train_digits, y_test_digits = train_test_split(all_digits_pixels, all_digits_labels,
                                                                                test_size=0.3)

# initialize a theta array, one for every class
multiclass_thetas = np.zeros((10, 65))

for class_no in range(n_classes):
    current_label = class_no
    # Hint: convert the labels array to have only 1's at the current class_no
    # START ANSWER
    converted_y_train_digits = np.where(y_train_digits == current_label, 1, 0)
    multiclass_thetas[class_no] = train_theta(x_train_digits, converted_y_train_digits)

    y_train_digits

    multiclass_thetas

    # END ANSWER
    print("class_no: " + str(class_no))

print("first 3 parameters of every theta")
print(multiclass_thetas[:, :3])
