"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    mu = mixture.mu
    var = mixture.var
    p = mixture.p
    n,d = X.shape
    L = np.zeros((n,len(p)))
    summ = 0
    for i in range(n):
        for j in range(len(p)):
            sq_sum = np.sum((X[i]-mu[j])**2)
            ex_part = np.exp((-1/(2*var[j]))*sq_sum)
            L[i][j] = p[j]*(1/(2*np.pi*var[j])**(d/2))*ex_part
        sum_ = np.sum(L[i])
        L[i] = L[i]/sum_
        summ = summ + np.sum(L[i]*np.log(sum_)) 
    return L,summ   
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,K = post.shape
    n,d = X.shape    
    p =np.ones(K)/K
    mu = X[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    mixture = GaussianMixture(mu,var,p)
    N = np.zeros(K)
    for i in range(K):
        N[i] = np.sum(post[:,i])
        mixture.p[i] = N[i]/n
        mixture.mu[i] = 1/(N[i])*(np.matmul(post[:,i],X))
        mixture.var[i] = 1/(N[i]*d)*(np.sum(post[:,i].reshape(n,1)*(X-mixture.mu[i])**2))
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
    mixt = mstep(X,L)
    L,l_new = estep(X,mixt)
    while abs((l_older-l_new)/l_new) >= 1E-6:
        l_older = l_new
        mixt = mstep(X,L)
        L,l_new = estep(X,mixt)
    return mixt,L,l_new
    raise NotImplementedError
