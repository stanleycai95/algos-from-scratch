import numpy as np

class KMeans:
    
    def __init__(self, n_clusters=10, tol=0.001, n_iter=50, n_init=5):
        self.n_clusters = n_clusters
        self.tol = tol
        self.n_iter = n_iter
        self.n_init = n_init
        self.all_labels = []
        
    def scale_data(self, X):
        self.X_max, self.X_min = X.max(axis=0), X.min(axis=0)
        X_scaled = (X - self.X_min) / np.where(self.X_max - self.X_min > 0, self.X_max - self.X_min, 1)
        
        return X_scaled
    
    def init_centroids(self, X):
        assert X.shape[0] >= self.n_clusters, "Less data points than clusters"
        
        self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]
        
    def euclidean_distance(self, X, centroid):
        assert X.shape[1] == centroid.shape[0], "Centroid does not have same number of features as data"
        
        distances = np.sqrt(np.sum(np.square(X - centroid), axis=1))
        
        return distances
    
    def fit(self, X):
        X = self.scale_data(X)
        
        for initialization in range(self.n_init):
            self.init_centroids(X)
            
            for iteration in range(self.n_iter):
                distance_to_centroids = np.hstack([self.euclidean_distance(X, centroid)[:,None] for centroid in self.cluster_centers_])
                self.labels = np.argmin(distance_to_centroids, axis=1)
                new_cluster_centers = np.hstack([X[self.labels == i].mean(axis=0)[:,None] for i in range(self.n_clusters)]).T
                if 1 - np.mean(new_cluster_centers == self.cluster_centers_) < self.tol:
                    break
                else:
                    self.cluster_centers_ = new_cluster_centers
            
            print(f"Total iterations: {iteration+1}")
            self.cluster_centers_ = self.cluster_centers_[np.lexsort(tuple([self.cluster_centers_[:,i] for i in range(self.cluster_centers_.shape[1])]))]
            distance_to_centroids = np.hstack([self.euclidean_distance(X, centroid)[:,None] for centroid in self.cluster_centers_])
            self.labels = np.argmin(distance_to_centroids, axis=1)
            self.all_labels.append(self.labels)
        
        return self.all_labels


def test_class():
  from sklearn.datasets import load_digits

  data, labels = load_digits(return_X_y=True)
  (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

  kmeans = KMeans()

  all_labels = np.vstack(kmeans.fit(data)).T

  print(all_labels[all_labels[:,0] == 0])
