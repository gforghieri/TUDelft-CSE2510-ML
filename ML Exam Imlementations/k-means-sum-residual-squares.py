import numpy as np

# We have a set of points and a set of initial cluster centroids.
# Implement a function that calculates the average sum of residual squares of the given clusters.
#
# In the solution template, the points and cluster centroids, are given. The average sum of residual squares (srs) should be returned.

class Solution():

    def solution(points, clusters):
        n = len(points)

        square_dist = np.zeros([len(points),len(clusters)])

        for j, p_j in enumerate(points):
            for i, c_i in enumerate(clusters):
                square_dist[j,i] = np.sum((c_i - p_j)*(c_i - p_j))

        min_square = np.min(square_dist, axis=1)

        srs = np.sum(min_square)/n
        return srs
