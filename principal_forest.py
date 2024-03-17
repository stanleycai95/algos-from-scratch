"""
Hypothesis:
One of the core principles behind Random Forest is the trees have to be decorrelated.
Why not decorrelate the features using PCA first? Then the trees will be more decorrelated.
Experiments are also showing that then the probability of each feature being selected needs to be scaled by the variance of each PC.

Results:
Accuracy: 0.69 on Cleveland Heart Disease (vs. 0.59 for Random Forest)
Accuracy: 0.97 on Banknote Dataset (vs. 0.73 for Random Forest)
"""

from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PrincipalForest:
    
    def __init__(self, method='C4.5', depth=5, num_trees=100):
        self.method = method
        self.depth = depth
        self.num_trees = num_trees
        self.decision_trees = []
        
    def build_tree(self):
        row_subset = np.random.choice(self.potential_samples, self.X.shape[0], replace=True) 
        dt = DecisionTree()
        dt.fit(X=self.X[row_subset,:], y=self.y[row_subset], 
               num_features=self.num_features_per_split, p=self.sk_pca.explained_variance_ratio_)
        
        return dt

    def fit(self, X, y):
        self.sk_pca = PCA(X.shape[1]).fit(X)
        self.std_scaler = StandardScaler().fit(X)
        self.X = self.sk_pca.transform(self.std_scaler.transform(X))
        self.y = y
        self.num_features_per_split = np.around(np.sqrt(self.X.shape[1])).astype(int)
        self.potential_samples = np.array(list(range(self.X.shape[0])))

        self.decision_trees = Parallel(n_jobs=-1)(delayed(self.build_tree)() for _ in range(self.num_trees))

    def predict(self, X):
        assert len(self.decision_trees) > 0, "Random Forest must be fit before prediction"
        
        X = self.sk_pca.transform(self.std_scaler.transform(X))
        ans = np.zeros(X.shape[0])
        for dt in self.decision_trees:
            ans += dt.predict_proba(X)
        
        ans /= len(self.decision_trees)
        
        ans[ans<=0.5] = 0
        ans[ans>0.5] = 1
        
        return ans
