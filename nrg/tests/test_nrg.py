'''
Tests for nrg.
'''
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
from scipy.linalg import eigh, eigvalsh
import pdb
import time

from nrgmap import quickmap, get_wlist
from pymps.blockmarker import SimpleBMG
from pymps.spaceconfig import SpaceConfig, SuperSpaceConfig
from pymps.toolbox.spin import sz, sx, sy
from pymps.toolbox.utils import quicksave, quickload

from ..impurity import *
from ..impuritymodel import *
from ..nrg import *
from ..lib import *


class NRGTest(object):
    '''
    test for nrg solving a chain.
    '''

    def __init__(self):
        pass

    def get_imp_bath(self, which='flat'):
        rhofunc, imp = self.get_instance(which)
        wlist = get_wlist(w0=1e-15, Nw=10000, mesh_type='log',
                          Gap=0.1 if which == 'sc' else 0, D=1.)
        chain = quickmap(wlist, rhofunc, Lambda=3., nsite=55)[0]
        return imp, chain

    def get_instance(self, which):
        if which == 'flat':
            U = 1e-3
            Gamma = U / 12.66 / pi
            #rhofunc=lambda w:0 if abs(w)>1. else Gamma/pi

            def rhofunc(w): return zeros([2, 2]) if abs(
                w) > 1. else (Gamma / pi * identity(2))
            impurity = AndersonImp(U=U, ed=-U / 2.)
        elif which == 'sc':
            rhofunc = get_dfunc_skewsc(
                Gap=0.1, Gamma=0.5 / pi, D=1., eta=1e-15, skew=1.)
            impurity = SC2Anderson(U=1.0, ed=-0.5)
        return rhofunc, impurity

    def test_chainopc(self):
        '''test for function construct_chainopc'''
        imp, bath = self.get_imp_bath('flat')
        scaledchain = ImpurityModel(impurity=imp, bath=bath)
        opc = scaledchain.get_opc()
        opc.show()
        pdb.set_trace()

    def test_nrg(self):
        '''
        test for nrg iteration.
        '''
        Lambda = 3.
        imp, bath = self.get_imp_bath('flat')
        bmg = SimpleBMG(spaceconfig=imp.spaceconfig, qstring='QM')
        TL = sqrt(Lambda**(-arange(bath.nsite + 1) - 2)) / 0.7
        savetxt('TL.dat', TL)

        # empty run
        res = NRGSolve(impurity=NullImp(), Lambda=sqrt(
            Lambda), bath=bath, bmg=bmg, maxN=500)
        EL0, expander0, bms0 = res['EL'], res['expander'], res['bms']
        mps0 = expander0.get_mps(bmg=bmg, bms=bms0)
        quicksave('mps0.dat', mps0)
        quicksave('EL0.dat', EL0)

        # loaded run
        res = NRGSolve(imp, bath, bmg=bmg, Lambda=sqrt(
            Lambda), maxN=500, call_back=None)
        EL, expander, bms = res['EL'], res['expander'], res['bms']
        mps = expander.get_mps(bmg=bmg, bms=bms)
        quicksave('mps.dat', mps)
        quicksave('EL.dat', EL)

    def test_tchi(self):
        '''Test Tchi'''
        EL0 = quickload('EL0.dat')
        EL = quickload('EL.dat')
        TL = loadtxt('TL.dat')
        mps = quickload('mps.dat')
        mps0 = quickload('mps0.dat')

        # calculate chi**2
        Szs = Tscale_qnumber(mps=mps, EL=EL, TL=TL)[1:, 1] / 2. -\
            Tscale_qnumber(mps=mps0, EL=EL0, TL=TL[1:])[:, 1] / 2.
        Sz2s = Tscale_qnumber(mps=mps, EL=EL, TL=TL, qfunc=lambda x: x**2)[1:, 1] / 4. -\
            Tscale_qnumber(mps=mps0, EL=EL0,
                           TL=TL[1:], qfunc=lambda x: x**2)[:, 1] / 4.
        ion()
        res = smear_evenodd(TL[1:], Sz2s - Szs**2)
        plot(-log(TL[1:-1]), res[:-1])
        pdb.set_trace()


# ChainTest().test_chainopc()
# NRGTest().test_nrg()
NRGTest().test_tchi()
