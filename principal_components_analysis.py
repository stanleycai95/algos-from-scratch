import numpy as np
import seaborn as sns
import pandas as pd

class PrincipalComponentsAnalysis:
    
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def get_covariance_matrix(self):
        self.X_means = np.mean(self.X, axis=0)
        X_centered = (self.X - self.X_means)
        self.X_cov = (X_centered.T @ X_centered)
        
    def get_eigenvalues(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.X_cov)
    
    def get_principal_components(self, k=2):
        assert np.all(np.abs(self.eigenvalues[:-1]) >= np.abs(self.eigenvalues[1:])), "eigenvalues not sorted descending"
        
        self.W = self.eigenvectors[:,:k]
    
    def get_dimensionality_reduction(self):
        self.X_dimension_reduction = self.X @ self.W
        
        return self.X_dimension_reduction
    
    def get_PCA(self, k=2, plotting=True):
        self.get_covariance_matrix()
        self.get_eigenvalues()
        self.get_principal_components(k)
        self.get_dimensionality_reduction()
        
        if plotting:
            self.X_df = pd.DataFrame(np.hstack((self.X_dimension_reduction, self.y[:,None])))
            self.X_df.columns = ['PC1', 'PC2', 'label']
            sns.scatterplot(data=self.X_df, x='PC1', y='PC2', hue='label')
    
    def get_abs_reconstruction_error(self):
        self.X_reconstructed = self.X_dimension_reduction @ self.W.T
        return np.mean(np.abs(self.X_reconstructed - self.X))
    
    
    
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X, y = cancer.data, cancer.target

pca = PrincipalComponentsAnalysis(X, y)
pca.get_PCA()
print(pca.get_abs_reconstruction_error())
print(pca.W)

from sklearn.decomposition import PCA
sk_pca = PCA(2).fit(X)
print(sk_pca.components_.T)
