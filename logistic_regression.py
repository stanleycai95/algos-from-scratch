import numpy as np

class LogisticRegression:
    
    def __init__(self, C=1e-3, l1_ratio=0, tol=1e-3):
        assert C > 0, "invalid C"
        assert 0 <= l1_ratio <= 1, "invalid l1_ratio"
        
        self.l1_reg = l1_ratio / C
        self.l2_reg = (1 - l1_ratio) / C
        self.tol = tol
        
    def reformat_y(self, y):
        if len(y.shape) == 1:
            y = y[:, None]
        return y
    
    def reformat_X(self, X):
        if np.sum(X[:,0]) != X.shape[0]:
            X_standardized = (X - np.mean(X, axis=0).T) / X.std(axis=0).T
            X_with_intercept = np.hstack((np.ones(X_standardized.shape[0])[:,None], X_standardized))
        else:
            X_with_intercept = X
        
        return X_with_intercept
        
    def fit(self, X, y, method='SGD', iterations=1000, lr=3e-3):
        X, y = self.reformat_X(X), self.reformat_y(y)
        self.Beta = np.random.normal(0, 0.01, (X.shape[1], 1))
        
        if method == 'SGD':
            for i in range(iterations):
                lr *= 0.99
                y_pred = self.predict(X)
                gradient = np.mean(X.T @ (y_pred - y), axis=1)[:,None]
                gradient_with_regularization = gradient + self.l1_reg * np.sign(self.Beta) + self.l2_reg * self.Beta
                self.Beta -= lr * gradient_with_regularization
                self.Beta[np.abs(self.Beta) < self.tol] = 0
    
    def sigmoid(self, X) :
        return 1 / (1 + np.exp(-X))
    
    def predict(self, X):
        X = self.reformat_X(X)
        y_pred = self.sigmoid(X @ self.Beta)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return y_pred_clipped
    
    def binary_crossentropy(self, y_pred, y):
        y = self.reformat_y(y)
        cost_vector = -(y * np.log(y_pred) + (1-y) * np.log(1 - y_pred))
        return np.mean(cost_vector)
    
    def score(self, y_pred, y):
        y = self.reformat_y(y)
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        return np.mean(y == y_pred)
    
    
def test_ml_algo_class()():
    import seaborn as sns

    titanic = sns.load_dataset('titanic')
    titanic = titanic.sample(frac=1).dropna()
    titanic_numeric = titanic.select_dtypes(include=np.number)
    X, y = titanic_numeric.drop(columns='survived').values, titanic_numeric['survived'].values
    X = (X - np.mean(X, axis=0).T) / X.std(axis=0).T

    train_test_cutoff = X.shape[0] * 4 // 5

    X_train, y_train = X[:train_test_cutoff,:], y[:train_test_cutoff]
    X_test, y_test = X[train_test_cutoff:,:], y[train_test_cutoff:]

    logreg = LogisticRegression(C, l1_ratio)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(logreg.score(y_pred, y_test))
    print(logreg.Beta.flatten())

    import sklearn as sk
    clf = sk.linear_model.LogisticRegression(random_state=0).fit(X_train, y_train)
    clf.predict(X_test)
    print(clf.score(X_test, y_test))
    print(clf.intercept_, clf.coef_)
