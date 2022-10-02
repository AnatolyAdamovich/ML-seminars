from cv2 import CAP_PVAPI_PIXELFORMAT_BAYER16
import numpy as np


class AgglomerativeAlgorithm:
    def __init__(self, method="single", metric="euclidean", number_of_clusters=False):
        """
        Parameters:
            - method: ["single", "average", "complete", "centroid", "ward"] 
                The method that would be used to calculate the distances between two clusters
            - metric: ["euclidean", "chebyshev", "manhattan"] 
                The metric that would be used to calculate the distances between each pair of elements of
                for initial distances matrix
            - number_of_clusters: boolean or integer
                set the number of clusters ()
        """
        # method initializing
        if method == "single":
            self.method = self.single_linkage
        elif metric == "complete":
            self.method = self.complete_linkage
        elif metric == "average":
            self.method = self.average_linkage
        elif metric == "centroid":
            self.method = self.centroid_linkage
        elif metric == "ward":
            self.method = self.ward_linkage
        else:
            raise ValueError(f"There is no method called '{method}' ")
        
        # metric initializing
        if metric == "euclidean":
            self.metric = self.euclidean_distance
        elif metric == "chebyshev":
            self.metric = self.chebyshev_distance
        elif metric == "manhattan":
            self.metric = self.manhattan_distance
        else:
            raise ValueError(f"There is no metric called '{metric}' ")

        # structure for clusters
        self.clusters = dict()


    ######################################### distances
    @staticmethod
    def euclidean_distance(vec1, vec2):
        return np.sqrt(np.sum((vec1-vec2)**2))
    @staticmethod
    def manhattan_distance(vec1, vec2):
        return np.sum(np.abs(vec1-vec2))
    @staticmethod
    def chebyshev_distance(vec1, vec2):
        return np.max(np.abs(vec1-vec2))

    
    def single_linkage(self, c1, c2, new_c, c):
        dist = min(self.clusters[c1]["distances"][c], self.clusters[c2]["distances"][c])
        self.clusters[new_c]["distances"][c] = dist
        self.clusters[c]["distances"][new_c] = dist
    
    def complete_linkage(self, c1, c2, new_c, c):
        dist = min(self.clusters[c1]["distances"][c], self.clusters[c2]["distances"][c])
        self.clusters[new_c]["distances"][c] = dist
        self.clusters[c]["distances"][new_c] = dist
    
    
    def average_linkage():
        pass

    
    def centroid_linkage():
        pass

    
    def ward_linkage():
        pass

    ################################# methods for algorithm
    def initial(self, data):
        """
        create initial distance matrix using data, defined metric; 
        also the structure for clusters is created here;
        """
        for i in range(len(data)):
            # initial distance matrix: count distance for each pair (counted twice now)
            self.clusters[i] = {
                "cluster_length": 1,      # the number of element in the cluster
                "cluster": [i],           # the cluster itself
                "distances": dict((j, self.metric(data[i], data[j])) for j in range(0, len(data)))
            }
            del self.clusters[i]['distances'][i]


    def find_min_distance(self):
        """
        function to find the minimum element in the distances matrix;
        find 2 minimum (first for merging two clusters, second for optimization)
        """
        min_dist = 1e6
        first_pair_of_clusters = ()
        for i in self.clusters:
            for j in self.clusters[i]["distances"]:
                if self.clusters[i]["distances"][j] < min_dist:
                    min_dist = self.clusters[i]["distances"][j]
                    pair_of_clusters = (i, j)
        return pair_of_clusters, min_dist
    
    
    def linkage(self, u, v, new_index):
        # w = u || v - new cluster
        # distance is calculated by formula:
        # R(w, s) = alpha 
        if self.method == "single":            
            for s in self.clusters:
                if s not in [u, v, new_index]:
                    self.clusters[s][new_index] = min(self.clusters[u][s], self.clusters[v][s])
                    self.clusters[new_index][s] = self.clusters[s][new_index]

        elif self.method == "complete":
             for s in self.clusters:
                if s not in [u, v, new_index]:
                    self.clusters[s][new_index] = max(self.clusters[u][s], self.clusters[v][s])
                    self.clusters[new_index][s] = self.clusters[s][new_index]

        elif self.method == "average":
            # future
            pass
        elif self.method == "centroid":
            # future
            pass
        elif self.method == "word":
            # future
            pass

        else:
            return 


    def merge_two_clusters(self, c1, c2, new_c):
        """
        function to create a new cluster (new_c) based on two existing clusters (c1, c2)
        """
        # firstly, create the new cluster
        self.clusters[new_c] = {
            "cluster_length": self.clusters[c1]["length"] + self.clusters[c2]["length"], 
            "cluster": self.cluster[c1]["cluster"] + self.cluster[c2]["cluster"],
            "distances": 0
            }

        
        # secondly, count the distance to new cluster from other
        self.linkage(c1, c2, new_c)
        
        # and finally, remove old clusters
        self.clusters.pop(c1, None)
        self.clusters.pop(c2, None)

        for c in self.clusters:
            self.clusters[c]["distances"].pop(c1, None)
            self.clusters[c]["distances"].pop(c2, None)


    def fit(self, X):
        """
        function corresponding to the main algorithm
        """
        n = len(X)
        self.initial(X)
        # linkage matrix is need for visualizing
        linkage_matrix = np.zeros([n-1, 4])

        # iterations
        for t in range(0, n-1):
            pair, dist = self.find_min_distance()
            u, v = pair
            self.merge_two_clusters(u, v, n+t)
            # fill the linkage matrix according to the requirements: 
            linkage_matrix[t][0] = u
            linkage_matrix[t][1] = v
            linkage_matrix[t][2] = dist
            linkage_matrix[t][3] = self.clusters[n+t][-1]

        return linkage_matrix