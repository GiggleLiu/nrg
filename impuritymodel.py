from scipy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import scipy.sparse as sps
import pdb,time,copy

from tba.hgen import H2G,SuperSpaceConfig,op_from_mats,op_on_bond,op_U
from tba.lattice import Bond
from nrgmap import construct_tridmat,Chain,show_scaling
from rglib.mps import op2collection,insert_Zs

__all__=['ImpurityModel','scale_bath']

def scale_bath(bath,Lambda):
    '''
    Rescale the bath by scaling factor Lambda.
    This will make the chain ballanced.
    
    Parameters:
        :bath: <Chain>,
        :Lambda: num/1d array, the scaling factor.

    Return:
        <Chain>, the chain after scaling.
    '''
    if ndim(Lambda)==0: Lambda=Lambda**arange(bath.nsite)
    factor=Lambda.reshape([-1]+[1]*(ndim(bath.t0)))
    elist=bath.elist*factor
    tlist=bath.tlist*factor[1:]
    ch=Chain(t0=bath.t0*factor[0],tlist=tlist,elist=elist)
    return ch

class ImpurityModel(object):
    '''
    The chain with scaling.
    '''
    def __init__(self,impurity,bath):
        self.impurity=impurity
        self.bath=bath

    @property
    def nsite(self):
        '''number of sites.'''
        if self.impurity.H0 is None:
            return self.bath.nsite
        return self.bath.nsite+1

    def get_H0(self):
        '''
        Hamiltonian without interaction terms.

        Return:
            sparse matrix, the hamiltonian.
        '''
        tlist=self.bath.tlist
        elist=self.bath.elist
        nz=elist.shape[1]
        if self.impurity.H0 is not None:
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

        #the new space configuration.
        nsite=self.nsite
        spaceconfig1=impurity.spaceconfig
        config=list(spaceconfig1.config)
        config[-2]=nsite
        spaceconfig=SuperSpaceConfig(config)

        elist,tlist=self.bath.elist,self.bath.tlist
        if impurity.H0 is not None:
            elist=concatenate([self.impurity.H0[newaxis,...],elist],axis=0)
            tlist=concatenate([self.bath.t0[newaxis,...],tlist],axis=0)
        tlist2=swapaxes(tlist.conj(),-1,-2)
        ebonds=[Bond(i,i,zeros(2)) for i in xrange(len(elist))]
        tbonds=[Bond(i,i+1,[1.,0]) for i in xrange(len(tlist))]
        tbonds2=[-b for b in tbonds]
        opc=op_on_bond('E',spaceconfig=spaceconfig,mats=elist,bonds=ebonds)
        opc=opc+op_on_bond('T',spaceconfig=spaceconfig,mats=concatenate([tlist2,tlist],axis=0),bonds=tbonds2+tbonds)
        opc.label='H'
        opc+=impurity.get_interaction()
        opcl=op2collection(opc)
        insert_Zs(opcl,spaceconfig1)
        return opcl

