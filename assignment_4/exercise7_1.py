import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


import time

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Logistic Regression"]

classifiers = [
    KNeighborsClassifier(3),
    LogisticRegression(multi_class='multinomial', solver='lbfgs')]

x7, y7 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
x7 += 2 * rng.uniform(size=x7.shape)
linearly_separable = (x7, y7)

ds = make_moons(100, noise=0.3, random_state=0,)
figure = plt.figure(figsize=(20, 5))
i = 1
n_iterations = 10
for iteration in range(n_iterations):
    # preprocess dataset, split into training and test part
    x7, y7 = ds
    x7 = StandardScaler().fit_transform(x7)
    x7_train, x7_test, y7_train, y7_test = train_test_split(x7, y7, test_size=.9, random_state=int(time.perf_counter()) + iteration)

    x7_min, x7_max = x7[:, 0].min() - .5, x7[:, 0].max() + .5
    y7_min, y7_max = x7[:, 1].min() - .5, x7[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x7_min, x7_max, h), np.arange(y7_min, y7_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # iterate over classifiers
    c = 0
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(classifiers), n_iterations, i+c*n_iterations)
        clf.fit(x7_train, y7_train)
        score = clf.score(x7_test, y7_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]-0.5

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, alpha=.8, levels=[0.5])

        # Plot the training points
        ax.scatter(x7_train[:, 0], x7_train[:, 1], c=y7_train, cmap=cm_bright,
                   edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        c+=1
    i += 1

plt.tight_layout()
plt.show()