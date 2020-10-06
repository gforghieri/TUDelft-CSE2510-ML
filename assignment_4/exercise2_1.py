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