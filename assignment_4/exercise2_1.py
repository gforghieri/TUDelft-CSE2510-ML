from exercise1_2 import *

# width: average of the column-wise max values
widths = np.zeros(len(binary_digits_images))
# START ANSWER
max_per_column = np.zeros(binary_digits_images.shape[2])
for i in range(len(widths)):
    for k in range(binary_digits_images.shape[2]):
        max_per_column[k] = np.max(binary_digits_images[i][:, k])
    widths[i] = np.average(max_per_column)
# END ANSWER

# length: average of the row-wise max values
lengths = np.zeros(len(binary_digits_images))
# START ANSWER
max_per_row = np.zeros(binary_digits_images.shape[2])
for i in range(len(lengths)):
    for k in range(binary_digits_images.shape[2]):
        max_per_row[k] = np.max(binary_digits_images[i][k, :])
    lengths[i] = np.average(max_per_row)
# END ANSWER

assert (widths[:5] == np.array([8.5, 8.0, 9.25, 8.125, 9.5])).all()
assert (lengths[:5] == np.array([12.875, 15.625, 15.0, 15.75, 14.875])).all()

binary_digits_features = np.vstack((widths, lengths)).T
# print(binary_digits_features[:10])


def plot_scatter(features, labels, db_x=None, db_y=None):
    widths = features[:, 0]
    lengths = features[:, 1]

    # separate the 2 classes
    widths_0 = widths[labels == 0]
    lengths_0 = lengths[labels == 0]
    widths_1 = widths[labels == 1]
    lengths_1 = lengths[labels == 1]

    # Plot
    plt.scatter(widths_1, lengths_1, c='blue', label='ones')
    plt.scatter(widths_0, lengths_0, c='red', label='zeros')

    # Extra code to plot the decision boundary
    # You won't be using this right away
    if not (db_x is None or db_y is None):
        plt.plot(db_x, db_y, label="Decision_Boundary")

    plt.title('Digits')
    plt.xlabel('width')
    plt.ylabel('length')
    plt.xlim((widths.min() - 1, widths.max() + 1))
    plt.ylim((lengths.min() - 1, lengths.max() + 1))
    plt.legend(loc=3)
    # plt.show()


# plot_scatter(binary_digits_features, binary_digits_labels)