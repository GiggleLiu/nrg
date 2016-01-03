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
from nrg import NRGSolve
from rgobj import RGobj_Tchi

class SChainTest(object):
    '''
    test for scaled chain.
    '''
    def __init__(self,which='flat'):
        self.which=which
        self.set_rhofunc(which)
        N=45
        Lambda=3.0
        zl=[1.0]
        spaceconfig=SuperSpaceConfig([2,1,1])
        wlist=get_wlist(w0=1e-12,Nw=10000,mesh_type='log',Gap=0.1 if which=='sc' else 0,D=1.)
        #map it to a sun model
        tickers,discmodel=quick_map(wlist=wlist,rhofunc=self.rhofunc,N=N,z=zl,\
                tick_params={'tick_type':'adaptive','Lambda':Lambda})
        #map it to a Wilson chain
        self.baths=map2chain(discmodel,prec=5000)
        self.scales=[ticker2scale(tickers,N,z) for z in zl]
        evolutor=NullEvolutor(hndim=spaceconfig.hndim)
        expander=FermionHGen(spaceconfig=spaceconfig,evolutor=evolutor)

    def set_rhofunc(self,which):
        if which=='flat':
            U=1e-3
            Gamma=U/10/pi
            self.rhofunc=lambda w:zeros([2,2]) if abs(w)>1. else (Gamma/pi*identity(2))
            self.impurity=AndersonImp(U=U,ed=-U/2.)
        elif which=='sc':
            rhofunc=get_dfunc_skewsc(Gap=0.1,Gamma=0.5/pi,D=1.,eta=1e-15,skew=1.)
            self.impurity=SC2Anderson(U=1.0,ed=-0.5)

    def test_chainopc(self):
        '''test for function construct_chainopc'''
        ion()
        for bath,scale in zip(self.baths,self.scales):
            scaledchain=ScaledChain(impurity=self.impurity,bath=bath,scale=scale)
            opc=scaledchain.get_opc()
            opc.show()
            pdb.set_trace()

    def test_nrg(self):
        '''
        test for nrg iteration.
        '''
        tchiflows=[]
        tchiflows_ghost=[]
        good_number='N' if self.which=='sc' else 'NM'
        for bath,scale in zip(self.baths,self.scales):
            scaledchain=ScaledChain(impurity=self.impurity,bath=bath,scale=scale)
            pdb.set_trace()
            scaledchain_ghost=ScaledChain(impurity=NullImp(),bath=bath,scale=scale)
            evolutor_ghost,bms_ghost,elist_ghost=NRGSolve(scaledchain_ghost,good_number=good_number)
            evolutor,bms,elist=NRGSolve(scaledchain,good_number=good_number)
            Tchi=RGobj_Tchi(spaceconfig=self.impurity.spaceconfig,mode='s')
            tchiflows.append(Tchi.measure(bms,elist=elist,scaling_factors=scaledchain.scaling_factors,M_axis=good_number.index('M')))
            tchiflows_ghost.append(Tchi.measure(bms_ghost,elist=elist_ghost,scaling_factors=scaledchain_ghost.scaling_factors,M_axis=good_number.index('M')))
        tchiflow=mean(tchiflows,axis=0)
        tchiflow[1:]-=mean(tchiflows_ghost,axis=0)
        ion()
        cla()
        plot(tchiflow)
        pdb.set_trace()

    def check_scaling(self):
        '''check scaling.'''
        ion()
        for scale,bath in zip(self.scales,self.baths):
            scaledchain=ScaledChain(impurity=self.impurity,bath=bath,scale=scale)
            scaledchain.check_scaling()
        #scaling_factor=self.scale.get_scaling_factor(arange(self.chain.nsite+1))
        #elist=concatenate([[self.impurity.H0],self.chain.elist[:,0]],axis=0)
        #tlist=concatenate([self.chain.t0,self.chain.tlist[:,0]],axis=0)
        #elist=elist*scaling_factor[tuple([slice(None)]+[None]*(ndim(elist)-1))]
        #tlist=tlist*scaling_factor[tuple([slice(1,None)]+[None]*(ndim(elist)-1))]
        #plot(tlist[:,0,0])
        pdb.set_trace()

class ImpTest(object):
    '''
    Test for impurities.
    '''
    def test_H0(self):
        spaceconfig=SuperSpaceConfig([2,1,1])
        ed,Bz=0.2,0.3
        imp=AndersomImp(U=1.,ed=ed,Bz=Bz)
        assert_allcose(imp.H0,ed*identity(2)+Bz*sz)

SChainTest().check_scaling()
#SChainTest().test_chainopc()
SChainTest().test_nrg()
