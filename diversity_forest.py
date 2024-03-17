"""
Hypothesis:
So part of the reason Random Forests do well is because the trees have to be decorrelated.
Why stop at decorrelating through bootstrapping? Why not just add random noise to data used to train each tree?

Results:
Inconclusive so far.
"""

from joblib import Parallel, delayed # parallelize the trees

class DiversityForest:
    
    def __init__(self, method='C4.5', depth=5, num_trees=100):
        self.method = method
        self.depth = depth
        self.num_trees = num_trees
        self.decision_trees = []
        
    def build_tree(self):
        row_subset = np.random.choice(self.potential_samples, self.X.shape[0], replace=True)
        X_subset = self.X[row_subset,:]
        X_noised = X_subset + np.random.normal(0, self.X_std, size=X_subset.shape)
        dt = DecisionTree()
        dt.fit(X=X_noised, y=self.y[row_subset], num_features=self.num_features_per_split)

        return dt

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num_features_per_split = np.around(np.sqrt(self.X.shape[1])).astype(int)
        self.potential_samples = np.array(list(range(self.X.shape[0])))

        self.X_std = np.std(X, axis=0)

        self.decision_trees = Parallel(n_jobs=-1)(delayed(self.build_tree)() for _ in range(self.num_trees))

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
