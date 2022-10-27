import numpy as np


class DBSCAN_algorithm:
    """Realization of DBSCAN clustering algorithm.
    ----------
    epsilon : float, default=0.5
        The neighborhood length.

    min_samples : int, default=3
        The number of samples in a neighborhood for object to be considered
        as a *root point*. This number included the object itself.

    metric : {'euclidean', 'manhattan', 'chebyshev'}, default='euclidean'
        The metric to use when calculating distance between instances
    """
    def __init__(self, epsilon=0.5, min_samples=3, metric='euclidean'):
        self.eps = epsilon
        self.m = min_samples

        # metric initializing
        if metric == "euclidean":
            self.metric = self.euclidean_distance
        elif metric == "chebyshev":
            self.metric = self.chebyshev_distance
        elif metric == "manhattan":
            self.metric = self.manhattan_distance
        else:
            raise ValueError(f"There is no metric called '{metric}' ")


    def fit(self, data):
        # distance_matrix
        self._create_distance_matrix(data=data)

        current_label = -1
        result_labels = np.array([current_label] * len(data))
        unlabeled = set(np.arange(len(data)))

        while unlabeled:
            point_index = unlabeled.pop()
            neigh_for_point = self._neighborhood(point_index=point_index)

            if len(neigh_for_point) >= self.m:
                # the point is root
                current_label += 1
                self._create_cluster(neigh=neigh_for_point,
                                     current_label=current_label,
                                     result_labels=result_labels,
                                     unlabeled=unlabeled)

        return result_labels



    # matrix with distances between each pair of points
    def _create_distance_matrix(self, data):
        m = len(data)
        self.distance_matrix = np.zeros((m, m))
        for i, point in enumerate(data):

            for j in range(m):
                if (j >= i):
                    self.distance_matrix[i][j] = self.metric(point, data[j])
                else:
                    self.distance_matrix[i][j] = self.distance_matrix[j][i]



    # find neighborhood
    def _neighborhood(self, point_index):
        m = len(self.distance_matrix)
        neigh_for_point = [i for i in range(m)
                           if self.distance_matrix[point_index][i] <= self.eps]
        return np.array(neigh_for_point)

    # cluster creating
    def _create_cluster(self, neigh, current_label, result_labels, unlabeled):
        result_labels[neigh] = current_label
        cluster = set(neigh)
        while cluster:
            elem = cluster.pop()
            neigh_for_elem = self._neighborhood(point_index=elem)
            if len(neigh_for_elem) >= self.m:
                for neighbor in neigh_for_elem:
                    if (result_labels[neighbor] != current_label) and (neighbor not in cluster):
                        cluster.add(neighbor)
                        result_labels[neighbor] = current_label
                        unlabeled.discard(neighbor)


    # distances between instances
    @staticmethod
    def euclidean_distance(vec1, vec2):
        return np.sqrt(np.sum((vec1-vec2)**2))

    @staticmethod
    def manhattan_distance(vec1, vec2):
        return np.sum(np.abs(vec1-vec2))

    @staticmethod
    def chebyshev_distance(vec1, vec2):
        return np.max(np.abs(vec1-vec2))