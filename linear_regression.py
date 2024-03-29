import numpy as np

class LinearRegression:
    
    def __init__(self, l1_penalty=0, l2_penalty=0):
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        
    def fit(self, X, y, method='analytical', lr=3e-1, iterations=100, tol=1e-4):
        X = np.hstack([np.ones(X.shape[0])[:,None], X])
        self.scale_params = (np.mean(X, axis=0), np.std(X, axis=0))
        self.scale_params[0][self.scale_params[0] == 1] = 0
        self.scale_params[1][self.scale_params[1] < 1] = 1
        X = (X - self.scale_params[0]) / self.scale_params[1]
        
        if method == 'analytical':
            self.Beta = np.dot(np.linalg.inv(np.dot(X.T, X) + \
                                             self.l2_penalty * np.identity(X.shape[1])), 
                               np.dot(X.T, y[:, None]))      
                
        elif method == 'subgradient':
            self.Beta = np.random.normal(0, 1, X.shape[1])
            for i in range(iterations):
                lr *= 0.99
                preds = np.dot(X, self.Beta)[:,None]
                gradient = np.mean(np.dot(X.T, preds - y[:, None]) / X.shape[0] + \
                                   self.l1_penalty * np.sign(self.Beta) + \
                                   self.l2_penalty * self.Beta, axis=1)
                self.Beta -= lr * gradient
                self.Beta[self.Beta < tol] = 0
            
            self.Beta = self.Beta[:,None]
            
        elif method == 'coordinate_descent':
            self.Beta = np.random.normal(0, 1, X.shape[1])
            self.alpha = self.l1_penalty * X.shape[0]
            for i in range(iterations // 10):
                for j in range(X.shape[1]):
                    p = np.dot(X.T[j,:], y - np.dot(np.delete(X, j, 1), np.delete(self.Beta, j, 0)))
                    z = np.dot(X.T[j,:], X[:,j])
                    if p > self.alpha: 
                        self.Beta[j] = (p - self.alpha) / z
                    elif p < -self.alpha:
                        self.Beta[j] = (p + self.alpha) / z
                    else:
                        self.Beta[j] = 0
            self.Beta = self.Beta[:,None]
        
        else:
            print("Error: invalid method")

    
    def predict(self, X):
        X = np.hstack([np.ones(X.shape[0])[:,None], X])
        X = (X - self.scale_params[0]) / self.scale_params[1]
        return np.dot(X, self.Beta)
    
    def mean_squared_error(self, preds, y):
        return np.dot((y[:,None] - preds).T, (y[:,None] - preds)) / y.shape[0]


def test_ml_algo_class():
    import seaborn as sns

    tips = sns.load_dataset("tips")
    sns.regplot(x="total_bill", y="tip", data=tips)
    tips = tips.sample(frac=1)
    X, y = tips[['total_bill', 'size']].values, tips['tip'].values

    X = np.hstack((X, np.random.normal(0, 30, X.shape[0])[:,None]))
    std_X = np.std(X, axis=0)
    std_X[std_X < 1] = 1
    X = (X - np.mean(X, axis=0)) / std_X

    train_test_cutoff = tips.shape[0]//5 * 4
    X_train, y_train = X[:train_test_cutoff,:], y[:train_test_cutoff]
    X_test, y_test = X[train_test_cutoff:,:], y[train_test_cutoff:]

    for l2_reg in [0]:
        for l1_reg in [0.1]:
            lm = LinearRegression(l1_penalty=l1_reg, l2_penalty=l2_reg)
            lm.fit(X_train, y_train, method='coordinate_descent')
            y_pred = lm.predict(X_test)
            print(f"MSE for l1 reg: {l1_reg}, l2_reg: {l2_reg}")
            print(lm.mean_squared_error(y_pred, y_test))
            print("Betas")
            print(lm.Beta.flatten())
            
    from sklearn.linear_model import ElasticNet
    regr = ElasticNet(random_state=0, alpha=1e-1)
    regr.fit(X_train, y_train)
    print("sklearn MSE")
    print(lm.mean_squared_error(regr.predict(X_test)[:, None], y_test))
    print("sklearn Betas")
    print(np.hstack(([regr.intercept_], regr.coef_)))
