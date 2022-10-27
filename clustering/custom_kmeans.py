import numpy as np
import matplotlib.pyplot as plt


class KMeansAlgorithm:
    """Implementation of K-means clustering algorithm

    Parameters
    -----------
    initialization : str, default="single"
                     The method that would be used to assign the first centroids
                     Available methods: "single", "++"
    metric : str, default="euclidean"
             The metric that would be used to calculate the distances
             between pair of elements
             Available metrics: "euclidean", "chebyshev", "manhattan"

    """
    def __init__(self, initialization="single", metric="euclidean"):
        # method initializing
        if initialization == "single":
            self.init = self.single_initialization
        elif initialization == "++":
            self.init = self.advanced_initialization
        else:
            raise ValueError(f"There is no initialization method called '{initialization}' ")

        # metric initializing
        if metric == "euclidean":
            self.metric = self.euclidean_distance
        elif metric == "chebyshev":
            self.metric = self.chebyshev_distance
        elif metric == "manhattan":
            self.metric = self.manhattan_distance
        else:
            raise ValueError(f"There is no metric called '{metric}' ")

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

    # methods for initialization
    @staticmethod
    def single_initialization(data, k):
        """Random selection of centroids"""
        index_centroids = np.random.choice(len(data), size=k, replace=False)
        centroids = data[index_centroids]
        return centroids

    @staticmethod
    def advanced_initialization(data, k):
        # kmeans++ method for init
        return np.array([])

    def fit(self, data, number_of_clusters, max_iter=100000, repeat=False, plot=False):
        if repeat:
            array_with_variations = []
            for n_iter in range(repeat):
                current_variation, current_labels, current_centroids = self._main_algorithm(data, number_of_clusters,
                                                                                            max_iter, plot)
                array_with_variations.append((current_variation, current_labels, current_centroids))
            result = min(array_with_variations, key=lambda element: element[0])
        else:
            result = self._main_algorithm(data, number_of_clusters, max_iter, plot)
        return result

    def _main_algorithm(self, data, number_of_clusters, max_iter, plot):
        labels = np.array([-1] * len(data))
        centroids = self.init(data=data, k=number_of_clusters)
        for i in range(max_iter):
            copy_centroids = centroids.copy()
            # assign instances with clusters
            self._objects_assignment(data, centroids, labels)
            # plot steps if need
            if plot:
                self.plot_steps(data, centroids, labels)
            # recalculation of centroids
            self._centroids_calculation(data, centroids, labels)
            if self.metric(copy_centroids, centroids) == 0:
                break
        variation = np.sum([self.metric(data[i], centroids[labels[i]])**2 for i in range(len(data))])
        return variation, labels.copy(), centroids.copy()

    def _objects_assignment(self, data, centroids, labels):
        for i in range(len(data)):
            element = data[i]
            distances = [self.metric(element, centroids[k]) for k in range(len(centroids))]
            labels[i] = np.argmin(distances)

    def _centroids_calculation(self, data, centroids, labels):
        for cluster_index in range(len(centroids)):
            cluster = data[labels == cluster_index]
            centroids[cluster_index] = np.mean(cluster, axis=0)

    def plot_steps(self, data, centroids, labels):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(data[:, 0], data[:, 1], c=labels, linewidths=1)
        for centroid in centroids:
            ax.scatter(*centroid, marker='x', color='m', linewidths=2)
        plt.show()

def plot_within_cluster_variation(data, model, repeat=3, range_of_clusters=None, figsize=(10, 4)):
    if range_of_clusters is None:
        range_of_clusters = np.arange(1, len(data))
    wcv_array = []
    for k in range_of_clusters:
        wcv, _, _ = model.fit(data, number_of_clusters=k, repeat=repeat)
        wcv_array.append(wcv)
    plt.figure(figsize=figsize)
    plt.plot(range_of_clusters, np.array(wcv_array), marker='o', linestyle='-', c='b')
    plt.ylabel('Within cluster variation')
    plt.xlabel('number of clusters')
    plt.show()


# dataset = np.array([[0, 0], [2, 4], [3, 3], [1, 2], [3, 0], [3, 1], [1, 1], [12, 18], [13, 17],
#                     [11, 15], [13, 14], [14, 16], [11, 16], [12, 15], [13, 18], [12, 5], [13, 2],
#                     [14, 4], [12, 3], [13, 1], [14, 2], [24, 19], [22, 22], [21, 24], [23, 21],
#                     [24, 20], [22, 39], [23, 38], [24, 39], [21, 37], [2, 26], [24, 6], [10, 36]])
# variation, labels, centroids = KMeansAlgorithm().fit(data=dataset, number_of_clusters=5, plot=True)