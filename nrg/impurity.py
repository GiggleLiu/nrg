import numpy as np

from tba.hgen import SuperSpaceConfig, sz, sx, sy
from tba.hgen import op_U, Operator


def anderson_impurity(U, mu, Bz=0.):
    h = np.diag([0, -mu+Bz, -mu-Bz, U-2*mu])
    return h

def empty_impurity():
    h = np.zeros([1,1])
    return h
