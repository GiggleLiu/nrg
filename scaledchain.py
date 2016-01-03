from scipy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import scipy.sparse as sps
import pdb,time,copy

from tba.hgen import H2G,SuperSpaceConfig,op_from_mats,op_on_bond,op_U
from tba.lattice import Bond
from nrgmap import construct_tridmat
from rglib.mps import op2collection
from impurity import NullImp

__all__=['ScaledChain']

class ScaledChain(object):
    '''
    The chain with scaling.
    '''
    def __init__(self,impurity,bath,scale):
        self.impurity=impurity
        self.bath=bath
        self.scale=scale

    @property
    def scaling_factors(self):
        '''the scaling factors.'''
        offset=0
        if isinstance(self.impurity,NullImp):
            offset=1
        return self.scale.get_scaling_factor(arange(offset,self.nsite+offset))

    @property
    def nsite(self):
        '''number of sites.'''
        if isinstance(self.impurity,NullImp):
            return self.bath.nsite
        return self.bath.nsite+1

    @property
    def elist_rescaled(self):
        '''The rescaled energy list.'''
        if isinstance(self.impurity,NullImp):
            elist=self.bath.elist
        else:
            elist=concatenate([[self.impurity.H0],self.bath.elist],axis=0)
        elist=elist*self.scaling_factors[:,newaxis,newaxis]
        return elist

    @property
    def tlist_rescaled(self):
        '''The rescaled hopping list.'''
        nsite=self.nsite
        if isinstance(self.impurity,NullImp):
            tlist=self.bath.tlist
        else:
            tlist=concatenate([self.bath.t0[newaxis],self.bath.tlist],axis=0)
        tlist=tlist*self.scaling_factors[1:,newaxis,newaxis]
        return tlist

    def get_H0(self):
        '''
        Hamiltonian without interaction terms.

        Return:
            sparse matrix, the hamiltonian.
        '''
        tlist=self.bath.tlist
        elist=self.bath.elist
        nz=elist.shape[1]
        if not isinstance(self.impurity,NullImp):
            elist=concatenate([self.impurity.H0[newaxis,...],elist],axis=0)
            tlist=concatenate([self.bath.t0[newaxis,...],tlist],axis=0)
        tlistH=swapaxes(tlist,1,2).conj()
        offset=[-1,0,1]
        data=[tlistH,elist,tlist]
        N=self.nsite
        B=ndarray([N,N],dtype='O')
        #fill datas
        for i in xrange(N):
            for j in xrange(N):
                for k in xrange(3):
                    if i-j==offset[k]:
                        B[i,j]=complex128(data[offset[k]+1][min(i,j)])
        return sps.bmat(B).toarray()

    def get_opc(self):
        '''
        Construct <OpCollection> for a chain.

        Return:
            <OpCollection>, the operator collection.
        '''
        impurity=self.impurity
        bath=self.bath

        nsite=bath.nsite+1
        spaceconfig1=impurity.spaceconfig
        config=list(spaceconfig1.config)
        config[-2]=nsite
        spaceconfig=SuperSpaceConfig(config)

        elist,tlist=self.elist_rescaled,self.tlist_rescaled
        tlist2=swapaxes(tlist.conj(),-1,-2)
        ebonds=[Bond(zeros(2),i,i) for i in xrange(len(elist))]
        tbonds=[Bond([1.,0],i,i+1) for i in xrange(len(tlist))]
        tbonds2=[Bond([-1.,0],i+1,i) for i in xrange(len(tlist))]
        opc=op_on_bond('E',spaceconfig=spaceconfig,mats=elist,bonds=ebonds)
        opc=opc+op_on_bond('T',spaceconfig=spaceconfig,mats=concatenate([tlist2,tlist],axis=0),bonds=tbonds2+tbonds)
        opc.label='H'
        opc+=impurity.get_interaction()
        return op2collection(opc).compactify()

    def check_scaling(self,**kwargs):
        '''
        Check the scaling behavior of this chain.
        '''
        tlist=self.tlist_rescaled
        elist=self.elist_rescaled
        tw=sqrt((tlist*tlist.conj()).sum(axis=(-1,-2)).real)
        ew=sqrt((elist*elist.conj()).sum(axis=(-1,-2)).real)
        x=arange(self.bath.nsite+1)
        plot(x[1:],tw,**kwargs)
        plot(x,ew,**kwargs)
        legend(['T','E'])
