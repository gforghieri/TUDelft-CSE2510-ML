from exercise1_1 import *

def plot_neighbours(X_train, Y_train, test_instance, k):
    """
    Plots all points in the dataset and shows the neighbours of a given test instance.
    """

    neighbours = get_neighbours(X_train, test_instance, k)
    # initialization of the sizes of the points to be plotted, size 10
    neigh_sizes = np.ones((len(Y_train), 1)) * 10
    neigh_sizes[neighbours] = 50
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=neigh_sizes)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.colorbar(ticks=[0, 1, 2], format=plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)]));
    plt.scatter(test_instance[0], test_instance[1], c='r', s=50, marker='x')
    plt.show()


for i in range(3):
    test_instance = X_test[i]
    k = 5
    plt.title('Test instance %s and its nearest neighbors' % (i + 1))
    plot_neighbours(X_train, Y_train, test_instance, k)