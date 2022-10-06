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
        self.n_of_clusters = number_of_clusters


    ######### distances between instances
    @staticmethod
    def euclidean_distance(vec1, vec2):
        return np.sqrt(np.sum((vec1-vec2)**2))
    
    @staticmethod
    def manhattan_distance(vec1, vec2):
        return np.sum(np.abs(vec1-vec2))
    
    @staticmethod
    def chebyshev_distance(vec1, vec2):
        return np.max(np.abs(vec1-vec2))

    ######### distances between clusters
    # 'new_c' is union of 'c1' and 'c2': new_c = c1 | c2
    # 'c' is the cluster to which we calculate the distance
    
    def single_linkage(self, c1, c2, new_c, c):
        """ R(new_c, c) = min(R(c1, c), R(c2, c)) """
        dist = min(self.clusters[c1]["distances"][c], self.clusters[c2]["distances"][c])
        self.clusters[new_c]["distances"][c] = dist
        self.clusters[c]["distances"][new_c] = dist
    
    def complete_linkage(self, c1, c2, new_c, c):
        """ R(new_c, c) = max(R(c1, c), R(c2, c)) """
        dist = max(self.clusters[c1]["distances"][c], self.clusters[c2]["distances"][c])
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
        create initial distance matrix using data, defined metric (distance function for objects); 
        also the structure for clusters is being created here;
        """
        for i in range(len(data)):
            # initial distance matrix: count distance for each pair (counted twice now)
            self.clusters[i] = {
                "cluster_length": 1,      # the number of element in the cluster
                "cluster": [i],           # the cluster itself
                "distances": dict((j, self.metric(data[i], data[j])) for j in range(i+1, len(data)))
            }
        for i in range(len(data)):
            for j in range(0, i):
                self.clusters[i]['distances'][j] = self.clusters[j]['distances'][i]


    def find_min_distance(self):
        """
        function to find the minimum element in the distances matrix;
        """
        # idea for optimization: find 2 minimum (first for merging two clusters, second for optimization)
        min_dist = 1e6
        for i in self.clusters:
            d = self.clusters[i]["distances"]
            index_for_min_in_current_cluster = min(d, key=d.get)
            if d[index_for_min_in_current_cluster] < min_dist:
                min_dist = d[index_for_min_in_current_cluster]
                pair_of_clusters = (i, index_for_min_in_current_cluster)
        return pair_of_clusters, min_dist
    
    
    def linkage(self, c1, c2, new_c):
        """
        function to calculate the distances between new cluster and others
        """
        for c in self.clusters:
            if c not in [new_c, c1, c2]:
                self.method(c1, c2, new_c, c)


    def merge_two_clusters(self, c1, c2, new_c):
        """
        function to create a new cluster (new_c) based on two existing clusters (c1, c2)
        """
        # firstly, create the new cluster
        self.clusters[new_c] = {
            "cluster_length": self.clusters[c1]["cluster_length"] + self.clusters[c2]["cluster_length"], 
            "cluster": self.clusters[c1]["cluster"] + self.clusters[c2]["cluster"],
            "distances": dict()
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
        # create initial clusters
        self.initial(X)
        
        # linkage matrix is need for visualizing
        linkage_matrix = np.zeros([n-1, 4])

        # iterations
        for t in range(0, n-1):
            pair, dist = self.find_min_distance()
            c1, c2 = pair

            self.merge_two_clusters(c1, c2, n+t)
            
            # fill the linkage matrix according to the requirements for creating dendrogram: 
            linkage_matrix[t][0] = c1
            linkage_matrix[t][1] = c2
            linkage_matrix[t][2] = dist
            linkage_matrix[t][3] = self.clusters[n+t]['cluster_length']
        
        return linkage_matrix