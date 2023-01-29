import numpy as np

class DecisionTree:
    
    def __init__(self, X, y, method='C4.5', depth=3):
        self.method = method
        self.depth = depth
        self.split_threshold = None
        self.split_attr = None
        self.split_info_gain = 0
        
        self.X = X
        self.y = y
        self.left = None
        self.right = None
        self.label = None
        self.fitted = False
        
    def check_valid_probability(self, p):
        assert np.isclose(np.sum(p), 1), "probabilities do not sum to 1"
        assert ((0 <= p) & (p <= 1)).all(), "probabilities are not between 0 and 1"        

    def get_proportions_from_labels(self, y):
        unique, counts = np.unique(y, return_counts=True)
        p = counts / np.sum(counts)        

        return p
        
    def gini_impurity(self, y):
        p = self.get_proportions_from_labels(y)   
        self.check_valid_probability(p)
        
        return 1 - np.sum(np.square(p))
    
    def entropy(self, y):
        p = self.get_proportions_from_labels(y)
        self.check_valid_probability(p)
        
        return -np.sum(p * np.log2(p))
    
    def accuracy(self, y_pred, y):
        return np.mean(y_pred == y)
    
    def fit(self):
        self.fitted = True
        
        if (self.depth == 0) or (np.mean(self.y) == 0) or (np.mean(self.y) == 1):
            self.label = (np.mean(self.y) > 0.5).astype(int)
        else:
            attributes = list(range(self.X.shape[1]))
                
            for attr in attributes:
                sort_order = self.X[:, attr].argsort()
                self.X = self.X[sort_order]
                self.y = self.y[sort_order]
                
                for i in range(self.X.shape[0]-1):
                    entropy_after_split = self.entropy(self.y[:i+1]) + self.entropy(self.y[i+1:])
                    entropy_before_split = self.entropy(self.y)
                    temp_info_gain = entropy_after_split - entropy_before_split
                    
                    if temp_info_gain < self.split_info_gain:
                        self.split_threshold = self.X[i, attr]
                        self.split_attr = attr
                        self.split_info_gain = temp_info_gain
                        
                        self.left = DecisionTree(X=self.X[:i+1,:], y=self.y[:i+1], method=self.method, depth=self.depth-1)
                        self.right = DecisionTree(X=self.X[i+1:,:], y=self.y[i+1:], method=self.method, depth=self.depth-1)

            if self.split_info_gain == 0:
                self.label = (np.mean(self.y) > 0.5).astype(int)
            else:
                self.left.fit()
                self.right.fit()
    
    def predict(self, X):
        assert self.fitted, "Decision tree needs to be fit before prediction"
        
        if self.label is not None:
            return np.array([self.label] * X.shape[0])
        else:
            ans = np.empty(X.shape[0])
            left_mask = X[:, self.split_attr] <= self.split_threshold
            right_mask = X[:, self.split_attr] > self.split_threshold
            
            ans[left_mask] = self.left.predict(X[left_mask])
            ans[right_mask] = self.right.predict(X[right_mask])
            
            return ans

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
shuffle = np.random.permutation(len(X))
X, y = X[shuffle], y[shuffle]

train_test_cutoff = X.shape[0] * 4//5
X_train, y_train = X[:train_test_cutoff,:], y[:train_test_cutoff]
X_test, y_test = X[train_test_cutoff:,:], y[train_test_cutoff:]

dt = DecisionTree(X_train, y_train)
dt.fit()
y_pred = dt.predict(X_test)

print(np.vstack((y_pred, y_test)).T)
print(dt.accuracy(y_pred, y_test))
print(np.mean(y))
