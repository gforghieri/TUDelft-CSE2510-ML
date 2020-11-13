import numpy as np
import sys


# This works when k = 2, that is 2 clusters will be created
def k_means_clustering_calculator(points, centroids):
    indices = []
    C1 = []
    C2 = []
    new_C1_coordinates = []
    new_C2_coordinates = []

    for point in points:
        distances = []
        for i in range(len(centroids)):
            distances.append(np.linalg.norm(point - centroids[i]))
        smallest_distance_index = np.argmin(distances)
        indices.append(smallest_distance_index)
        if smallest_distance_index == 0:
            C1.append(point)
        else:
            C2.append(point)

    array_c1 = np.array(C1)
    array_c2 = np.array(C2)

    new_C1_coordinates = np.mean(array_c1[:, 0]), np.mean(array_c1[:, 1])
    new_C2_coordinates = np.mean(array_c2[:, 0]), np.mean(array_c2[:, 1])

    return indices, C1, C2, new_C1_coordinates, new_C2_coordinates

# change centroids
centroids = np.array([[2, 8], [7, 2]])

#change data
data = np.array([[2,2], [8,6], [6,8], [2,4]])

results = k_means_clustering_calculator(data, centroids)
print("Cluster 1")
print(results[1])
print()
print("Cluster 2")
print(results[2])
print()
print("new C1 coordinates")
print(results[3])
print()
print("new C2 coordinates")
print(results[4])
