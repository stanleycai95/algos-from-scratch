import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    
    def __init__(self, method='C4.5', depth=5, num_trees=10):
        self.method = method
        self.depth = depth
        self.num_trees = num_trees
        self.decision_trees = []
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        num_features_per_split = np.around(np.sqrt(self.X.shape[1])).astype(int)
        potential_samples = np.array(list(range(self.X.shape[0])))
        
        for i in range(self.num_trees):
            row_subset = np.random.choice(potential_samples, self.X.shape[0], replace=True)
            
            dt = DecisionTree()
            dt.fit(X=self.X[row_subset,:], y=self.y[row_subset], num_features=num_features_per_split)
            self.decision_trees.append(dt)

    def predict(self, X):
        assert len(self.decision_trees) > 0, "Random Forest must be fit before prediction"
        
        ans = np.zeros(X.shape[0])
        for dt in self.decision_trees:
            ans += dt.predict_proba(X)
        
        ans /= len(self.decision_trees)
        
        ans[ans<=0.5] = 0
        ans[ans>0.5] = 1
        
        return ans
            
def accuracy(y_pred, y):
    return np.mean(y_pred == y)

def test_ml_algo_class():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    shuffle = np.random.permutation(len(X))
    X, y = X[shuffle], y[shuffle]

    train_test_cutoff = X.shape[0] * 4//5
    X_train, y_train = X[:train_test_cutoff,:], y[:train_test_cutoff]
    X_test, y_test = X[train_test_cutoff:,:], y[train_test_cutoff:]

    dt = DecisionTree()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    rf = RandomForest()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("Decision tree accuracy")
    print(accuracy(y_pred, y_test))

    print("Random Forest accuracy")
    print(accuracy(y_pred_rf, y_test))

    print("Class balance check")
    print(np.mean(y))
