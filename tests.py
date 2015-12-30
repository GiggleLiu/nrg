'''
Tests for nrg.
'''
from numpy import *
from matplotlib import pyplot as plt
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose

from nrgmap import quick_map,get_wlist,check_disc,map2chain,check_spec
from dfunclib import *
from impurity import *
from tba.hgen import SpaceConfig,SuperSpaceConfig
from tba.hgen.utils import sz,sx,sy
from rglib.hexpand import Evolutor,FermionHGen
from scaledchain import *
from scale import *
from rglib.hexpand import NullEvolutor 

class SChainTest(object):
    '''
    test for scaled chain.
    '''
    def __init__(self):
        Gap,D=0.1,1
        N=5
        Lambda=2.0
        z=1.0
        tick_type='adaptive'
        spaceconfig=SuperSpaceConfig([2,1,1])
        wlist=get_wlist(w0=1e-8,Nw=10000,mesh_type='sclog',Gap=Gap,D=D)
        rhofunc=get_dfunc_skewsc(Gap=Gap,Gamma=0.5/pi,D=D,eta=1e-15,skew=0)
        #map it to a sun model
        tickers,discmodel=quick_map(wlist=wlist,rhofunc=rhofunc,N=N,z=z,\
                tick_params={'tick_type':tick_type,'Lambda':Lambda})
        #map it to a Wilson chain
        self.chain=map2chain(discmodel,prec=3000)
        scale_ticks=[tickers[0](arange(N)+z),tickers[1](arange(N)+z)]
        self.scale=EScale(scale_ticks,Lambda,z,tick_type=tick_type)
        self.impurity=SCImp(U=1.0,ed=-0.5)
        evolutor=NullEvolutor(hndim=spaceconfig.hndim)
        expander=FermionHGen(spaceconfig=spaceconfig,evolutor=evolutor)

    def test_chainopc(self):
        '''test for function construct_chainopc'''
        opcs=construct_chainopc(impurity=self.impurity,chain=self.chain,scale=self.scale)
        ion()
        opcs[0].show()
        pdb.set_trace()

    def test_schain(self):
        pass

class ImpTest(object):
    '''
    Test for impurities.
    '''
    def test_H0(self):
        spaceconfig=SuperSpaceConfig([2,1,1])
        ed,Bz=0.2,0.3
        imp=AndersomImp(U=1.,ed=ed,Bz=Bz)
        assert_allcose(imp.H0,ed*identity(2)+Bz*sz)

SChainTest().test_chainopc()
