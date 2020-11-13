import numpy as np
import sys

# During this assignment you’re going to implement a part of the k means clustering algorithm.
# To be able to update the centroids of the clusters in the future, we need to know which points of the dataset belong to which centroid.
#
# Given are points (np.array) and centroids (np.array), implement a function that returns a list,
# that contains for each data point the index to its closest centroid. Use the euclidean distance for the distance computation.
# If it is a tie, your function should return the centroid with the lowest index.
#
# Hint: you can use np.linalg.norm(p1 - p2) to calculate the euclidean distance between two points.

class Solution():

    def solution(points, centroids):

        indices = []

        for p in points:
            mindist = sys.maxsize
            for i, c in enumerate(centroids):
                dist = np.linalg.norm(p - c)
                if dist < mindist:
                    mindist = dist
                    label_index = i
            indices.append(label_index)

        return indices
