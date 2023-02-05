import numpy as np
from decision_tree import DecisionTree

class GBDT:
    
    def __init__(self, num_iterations=10, scaling_param=0.5, depth=2, regression=True):
        self.num_iterations = num_iterations
        self.scaling_param = scaling_param
        self.regression = regression
        self.depth = depth
        self.fitted = False
    
    def fit(self, X, y):
        self.fitted = True
        self.intercept = np.median(y)
        self.models = []
        
        y_hat = np.ones(y.shape) * self.intercept
        curr_loss = loss_func(y_hat, y)
        
        for i in range(self.num_iterations):
            pseudo_residuals = (y - y_hat)
            dt = DecisionTree(depth=self.depth, regression=self.regression)
            dt.fit(X=X, y=pseudo_residuals)
            y_hat += self.scaling_param * dt.predict(X)
            print(loss_func(y_hat, y))
            if loss_func(y_hat, y) < curr_loss:
                curr_loss = loss_func(y_hat, y)
                self.models.append(dt)
            else:
                break
    
    def predict(self, X):
        assert self.fitted, "Decision tree needs to be fit before prediction"
        
        y_pred = np.ones(X.shape[0]) * self.intercept
        for i in range(len(self.models)):
            y_pred += self.scaling_param * self.models[i].predict(X)
        
        return y_pred
        
def loss_func(y_pred, y, regression=True):
    if regression:
        return np.mean(np.abs(y_pred - y))
    else:
        return 1 - np.mean(y_pred == y)

from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
shuffle = np.random.permutation(len(X))
X, y = X[shuffle], y[shuffle]

train_test_cutoff = X.shape[0] * 4//5
X_train, y_train = X[:train_test_cutoff,:], y[:train_test_cutoff]
X_test, y_test = X[train_test_cutoff:,:], y[train_test_cutoff:]

dt = DecisionTree(regression=True)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Decision tree loss")
print(loss_func(y_pred, y_test))

gbdt = GBDT(regression=True)
gbdt.fit(X_train, y_train)
y_pred_gbdt = gbdt.predict(X_test)

print("GBDT loss")
print(loss_func(y_pred_gbdt, y_test))
