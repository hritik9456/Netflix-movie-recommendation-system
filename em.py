"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n,d = X.shape
    K = mixture.mu.shape[0]
    lpi = np.log(mixture.p+1e-16) 
    N = np.zeros((n,K))
    N_copy = np.copy(N)
    N.astype('float64')
    sum2 = 0
    for i in range(n):
        nozero_X = X[i][np.nonzero(X[i])]
        d = len(nozero_X)
        for j in range(K):
            N[i][j] = lpi[j] + (-1/(2*mixture.var[j]))*np.sum((nozero_X-mixture.mu[j][np.nonzero(X[i])])**2) - (d/2)*(np.log(2*np.pi*mixture.var[j]))
        N_copy[i] = np.copy(N[i])
        N[i] = N[i] - logsumexp(N[i])
    sum2 = np.sum(N_copy[:,0]-N[:,0])  
    return np.exp(N),sum2         
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,d = X.shape
    n, K = post.shape
    sum_1 = np.sum(post,axis = 0)
    p = sum_1/n
    temp_var = np.zeros(K)
    mix_mu = np.zeros((K,d))
    mix_sum =np.zeros((K,d))
    mix_mu = np.matmul(post.transpose(),X)
    U = np.copy(X)
    U[U!=0] = 1
    mix_sum = np.matmul(post.transpose(),U)
    mix_sum[mix_sum==0] = 1
    for i in range(K):
        for j in range(d):
            if mix_sum[i][j] >=1 and mix_mu[i][j] != 0:
                mixture.mu[i][j] = mix_mu[i][j]/mix_sum[i][j]
    
    for j in range(K):
       sum_ = 0
       diff = 0
       for i in range(n):
           C_u = np.count_nonzero(X[i])  
           nozero_X =  X[i][np.nonzero(X[i])]
           sum_ = sum_ + C_u*post[i][j]
           diff = diff + post[i][j]*(np.sum((nozero_X - mixture.mu[j][np.nonzero(X[i])])**2))
       temp_var[j] = diff/sum_
       if temp_var[j] < 0.25:
           temp_var[j] = 0.25   
    mixture.var[:] = temp_var[:]
    mixture.p[:] = p[:]
    return mixture
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    L,l_older = estep(X,mixture)
    mixt = mstep(X,L,mixture)
    L,l_new = estep(X,mixt)
    while abs((l_new-l_older)/l_new) >= 1E-6:
        l_older = l_new
        mixt = mstep(X,L,mixt)
        L,l_new = estep(X,mixt)
    return mixt,L,l_new
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n,d1 = X.shape
    X_copy = np.copy(X)
    K = mixture.mu.shape[0]
    lpi = np.log(mixture.p+1e-16) 
    N = np.zeros((n,K))
    N.astype('float64')
    for i in range(n):
        nozero_X = X[i][np.nonzero(X[i])]
        #zero_X[i] = X[i][np.where(X[i]==0)]
        d = len(nozero_X)
        for j in range(K):
            #sq_sum = np.exp((-1/(2*mixture.var[j]))*np.sum((nozero_X-mixture.mu[j][np.nonzero(X[i])])**2))
            N[i][j] = lpi[j] + (-1/(2*mixture.var[j]))*np.sum((nozero_X-mixture.mu[j][np.nonzero(X[i])])**2) - (d/2)*(np.log(2*np.pi*mixture.var[j]))
        N[i] = np.exp(N[i] - logsumexp(N[i]))
        X_copy[i][np.where(X[i]==0)] = np.round(np.matmul(N[i],mixture.mu[:,np.where(X[i]==0)].reshape(K,d1-d)))    
    return X_copy
       
    raise NotImplementedError
