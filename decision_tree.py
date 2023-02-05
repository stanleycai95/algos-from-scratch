import numpy as np

class DecisionTree:
    
    def __init__(self, method='C4.5', depth=5, regression=False):
        self.method = method
        self.depth = depth
        self.split_threshold = None
        self.split_attr = None
        self.split_info_gain = 0
        self.left = None
        self.right = None
        self.label = None
        self.fitted = False
        self.regression = regression
        
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
    
    def coefficient_variation(self, y):
        return np.std(y) / max(abs(np.mean(y)), 1)
    
    def construct_splits(self, X):
        left_mask = X[:, self.split_attr] <= self.split_threshold
        right_mask = X[:, self.split_attr] > self.split_threshold
        
        return left_mask, right_mask
    
    def fit(self, X, y, num_attributes=None):
        self.fitted = True
        self.X, self.y = X, y
        
        self.num_attributes = num_attributes
        if self.num_attributes is None:
            self.num_attributes = self.X.shape[1]
        potential_features = np.array(list(range(self.X.shape[1])))
        self.features = np.random.choice(potential_features, self.num_attributes, replace=False)
        
        if not self.regression and ((self.depth == 0) or (np.mean(self.y) == 0) or (np.mean(self.y) == 1)):
            self.label = np.mean(self.y)
        elif self.regression and ((self.depth == 0) or (self.coefficient_variation(self.y) < 0.1)):
            self.label = np.mean(self.y)
        else:
            for attr in self.features:
                sort_order = self.X[:, attr].argsort()
                self.X = self.X[sort_order]
                self.y = self.y[sort_order]
                
                for i in range(self.X.shape[0]-1):
                    if not self.regression:
                        entropy_after_split = self.entropy(self.y[:i+1]) + self.entropy(self.y[i+1:])
                        entropy_before_split = self.entropy(self.y)
                        temp_info_gain = entropy_after_split - entropy_before_split
                    else:
                        cv_after_split = 1 / len(self.y) * ((i+1) * self.coefficient_variation(self.y[:i+1]) + (len(self.y) - i - 1) * self.coefficient_variation(self.y[i+1:]))
                        cv_before_split = self.coefficient_variation(self.y)
                        temp_info_gain = cv_after_split - cv_before_split
                    
                    if temp_info_gain < self.split_info_gain:
                        self.split_threshold = self.X[i, attr]
                        self.split_attr = attr
                        self.split_info_gain = temp_info_gain

            if self.split_info_gain == 0:
                self.label = np.mean(self.y)
            else:
                left_mask, right_mask = self.construct_splits(self.X)
                self.left = DecisionTree(method=self.method, depth=self.depth-1, regression=self.regression)
                self.right = DecisionTree(method=self.method, depth=self.depth-1, regression=self.regression)
                
                self.left.fit(X=self.X[left_mask], y=self.y[left_mask], num_attributes=self.num_attributes)
                self.right.fit(X=self.X[right_mask], y=self.y[right_mask], num_attributes=self.num_attributes)
    
    def predict_proba(self, X):
        assert self.fitted, "Decision tree needs to be fit before prediction"
        
        if self.label is not None:
            return np.array([self.label] * X.shape[0])
        else:
            ans = np.empty(X.shape[0])
            left_mask, right_mask = self.construct_splits(X)
            
            ans[left_mask] = self.left.predict(X[left_mask])
            ans[right_mask] = self.right.predict(X[right_mask])
            
            return ans
    
    def predict(self, X):
        y_hat = self.predict_proba(X)
        if self.regression:
            return y_hat
        else:
            return (y_hat > 0.5).astype(int)
