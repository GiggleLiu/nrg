import numpy as np

def anderson_impurity(U, mu, Bz=0.):
    h = np.diag([0, -mu+Bz, -mu-Bz, U-2*mu])
    return h

def empty_impurity():
    h = np.zeros([1,1])
    return h
