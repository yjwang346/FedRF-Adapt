#from sklearn.base import BaseEstimator
from turtle import forward
from scipy.stats import cauchy, laplace
#from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import torch
import os 
import torch.nn as nn

class RFF_perso(nn.Module):

    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.
    ----------
    sigma : float
         Parameter of RBF kernel: exp(-sigma * x^2)
    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.
    """
        
    def __init__(self, sigma=1., n_components=100, random_state=None, kernel = 'rbf'):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel = kernel
        self.sigma = sigma
        self.n_components = n_components
        self.random_state = random_state
        super(RFF_perso, self).__init__()

    def forward(self):
        pass

    def fit(self, feature_dim, y=None):
        """Fit the model with X.
        Samples random projection according to n_features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the transformer.
        """
        # self.random_offset_ = np.random.uniform(0, 2 * np.pi,
        #                                            size=self.n_components)
        
        # feature_dim = X.size(dim=1)
        if self.kernel == 'rbf':
            self.random_weights_ = (1.0 / self.sigma * torch.randn(size=(feature_dim, self.n_components)))
        elif self.kernel == 'laplacian':
            self.random_weights_ = laplace.rvs(scale = self.sigma, size=(feature_dim, self.n_components))
        elif self.kernel == 'cauchy':
            self.random_weights_ = cauchy.rvs(scale = self.sigma, size = (feature_dim, self.n_components))
        
        return self
    
    
    def transform(self,X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        # random_weights_ = self.random_weights_ # .to(self.device)
        sqet_n_components = torch.sqrt( torch.tensor(self.n_components)) # .to(self.device)

        device = X.device

        projection = torch.mm(X, self.random_weights_.to(device))
        pro_cos = torch.cos(projection)
        pro_sin = torch.sin(projection)
        projection = torch.cat((pro_cos, pro_sin), dim=1)
        projection *= 1.0 / sqet_n_components
        
        return projection
    
    def compute_kernel(self, X):
        """Computes the approximated kernel matrix.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        K : approximated kernel matrix
        """
        projection = self.transform(X)
        K = torch.mm(projection, projection.T)
        
        return K

    def rf_map(self):
        return self.random_weights_

        
class Sigma_items(nn.Module):
    def __init__(self, feature_dim, sigma=1., n_random_features=100, random_state=None, kernel = 'rbf'):
        super(Sigma_items, self).__init__()
        self.rf_map = RFF_perso(sigma, n_random_features, kernel=kernel)
        self.rf_map.fit(feature_dim)
        self.Sigma_flag = 0
        self.Sigma = torch.zeros(2,3)
        self.Sigma_1 = torch.zeros(2,3)
        self.Sigma_Sigma_T = torch.zeros(2,3)
        self.Sigma_y = torch.zeros(2,3)
        # self.y = torch.zeros(3)

    def calculate_items(self, X, ns, domain='source'):
        Xdevice = X.device
        ns_i = X.size(dim=0)

        X = torch.t(X)
        X = (X / (torch.linalg.norm(X, dim=0) + 1e-5))
        X = torch.t(X)

        assert not torch.any(torch.isnan(X))
        Sigma_i = self.rf_map.transform(X).T # Sigma_i: (2*n_features, n)
        Sigmai_1 = torch.mm(Sigma_i, torch.ones(ns_i, 1).to(Xdevice))
        Sigmai_11_SigmaiT = torch.mm(Sigmai_1, Sigmai_1.T)
        Sigmai_Sigmai_T = torch.mm(Sigma_i, Sigma_i.T)

        if domain == 'source':
            yi = 1/ns * torch.ones(ns_i, 1).to(Xdevice)
        elif domain == 'target':
            yi = -1/ns * torch.ones(ns_i, 1).to(Xdevice)
            
        Sigmai_yi = torch.mm(Sigma_i, yi)
        assert not torch.any(torch.isnan(Sigma_i))
        # put all together
        if self.Sigma_flag == 0:
            self.Sigma_flag = 1
            self.Sigma = Sigma_i
            self.Sigma_11_SigmaT = Sigmai_11_SigmaiT
            self.Sigma_Sigma_T = Sigmai_Sigmai_T
            self.Sigma_y = Sigmai_yi
        else:
            self.Sigma = torch.cat((self.Sigma, Sigma_i), dim=1)
            # self.Sigma_1 = self.Sigma_1 + Sigmai_1
            self.Sigma_11_SigmaT = torch.add(self.Sigma_11_SigmaT, Sigmai_11_SigmaiT)
            self.Sigma_Sigma_T = torch.add(self.Sigma_Sigma_T, Sigmai_Sigmai_T)
            self.Sigma_y = self.Sigma_y + Sigmai_yi
    
    def get_items(self):
        return self.Sigma, self.Sigma_11_SigmaT, self.Sigma_y, self.Sigma_Sigma_T
    
    def get_Sigmai(self, X):
        X = torch.t(X)
        X = (X / (torch.linalg.norm(X, dim=0) + 1e-5))
        X = torch.t(X)
        Sigma_i = self.rf_map.transform(X).T
        return Sigma_i