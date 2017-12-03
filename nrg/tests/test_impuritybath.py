from numpy import *
from matplotlib import pyplot as plt
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
import pdb
import time
import sys
sys.path.insert(0, '../')

from impurity import *


class ImpTest(object):
    '''
    Test for impurities.
    '''

    def test_H0(self):
        spaceconfig = SuperSpaceConfig([2, 1, 1])
        ed, Bz = 0.2, 0.3
        imp = AndersomImp(U=1., ed=ed, Bz=Bz)
        assert_allcose(imp.H0, ed * identity(2) + Bz * sz)
