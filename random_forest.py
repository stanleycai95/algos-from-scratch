import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    
    def __init__(self, X, y, method='C4.5', depth=5, num_trees=10):
        self.method = method
        self.depth = depth
        self.X = X
        self.y = y
        self.num_trees = num_trees
        self.decision_trees = []
        
    def fit(self):
        num_features_per_tree = np.around(np.sqrt(X.shape[1])).astype(int)
        potential_features = np.array(list(range(self.X.shape[1])))
        num_samples_per_tree = X.shape[0] * 2 // 3
        potential_samples = np.array(list(range(self.X.shape[0])))
        
        for i in range(self.num_trees):
            feature_subset = np.random.choice(potential_features, num_features_per_tree, replace=False)
            row_subset = np.random.choice(potential_samples, num_samples_per_tree, replace=False)
            
            dt = DecisionTree(X=X[row_subset,:], y=y[row_subset], attributes=feature_subset)
            dt.fit()
            self.decision_trees.append(dt)

    def predict(self, X):
        assert len(self.decision_trees) > 0, "Random Forest must be fit before prediction"
        
        ans = np.zeros(X.shape[0])
        for dt in self.decision_trees:
            ans += dt.predict(X)
        
        ans /= len(self.decision_trees)
        
        ans[ans<=0.5] = 0
        ans[ans>0.5] = 1
        
        return ans
            
def accuracy(y_pred, y):
    return np.mean(y_pred == y)

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

rf = RandomForest(X_train, y_train)
rf.fit()
y_pred_rf = rf.predict(X_test)

print("Decision tree accuracy")
print(accuracy(y_pred, y_test))

print("Random Forest accuracy")
print(accuracy(y_pred_rf, y_test))

print("Class balance check")
print(np.mean(y))
