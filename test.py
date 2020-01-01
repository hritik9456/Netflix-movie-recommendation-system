import numpy as np
import em
import common

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")
K = 12
n, d = X.shape
seed = 0
mixture, post=common.init(X,12,4)
mixture,L_s,l_m = em.run(X,mixture,post)
L = em.fill_matrix(X,mixture)
RMSE = common.rmse(X_gold,L)
print(RMSE)
# TODO: Your code here
