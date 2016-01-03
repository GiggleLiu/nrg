from scipy import *
from matplotlib.pyplot import *
import pdb,time

#from tdnrg import RGTimeLine
from scaledchain import ScaledChain
from rglib.hexpand import MaskedEvolutor,FermionHGen
from blockmatrix import get_bmgen,tobdmatrix
from utils import plot_spectrum
from impurity import NullImp

#MPI setting
try:
    from mpi4py import MPI
    COMM=MPI.COMM_WORLD
    SIZE=COMM.Get_size()
    RANK=COMM.Get_rank()
except:
    COMM=None
    SIZE=1
    RANK=0

def NRGSolve(scaledchain,good_number='NM',maxN=600,show_spec=True):
    '''
    Using NRG iteration method to solve a chain.

    Parameters:
        :chain: <ScaledChain>, the scaled chain for nrg.
        :good_number: str('N','NM','M'), the good quantum number.
        :maxN: integer, the maximum retained energy levels.
        :show_spec: bool, show spectrum during iteraction.

    Return:
        tuple of (evolutor, elist, bms), the <Evolutor>, energy of each iteraction, block marker of each iteraction.
        
        Note, elist is rescaled back.
    '''
    nsite=scaledchain.nsite
    is_ghost=isinstance(scaledchain.impurity,NullImp)
    scaling_factors=scaledchain.scaling_factors
    spaceconfig=scaledchain.impurity.spaceconfig
    h=scaledchain.get_opc()

    expander=FermionHGen(spaceconfig=spaceconfig,evolutor=MaskedEvolutor(hndim=spaceconfig.hndim))
    bmgen=get_bmgen(spaceconfig,good_number)
    elist=[]
    bms=[]
    for i in xrange(nsite):
        print 'Running iteraction %s'%(i+1)
        ops=h.filter(lambda sites:all(sites<=i) and (i in sites))
        H=expander.expand(ops)
        bm=bmgen.update_blockmarker(expander.block_marker,kpmask=expander.evolutor.kpmask(i-1))
        H_bd=bm.blockize(H)
        if not bm.check_blockdiag(H_bd):
            raise Exception('Hamiltonian is not block diagonal with good quantum number %s'%good_number)
        H=tobdmatrix(H_bd,bm)
        E,U=H.eigh()
        E_sorted=sort(E)
        kpmask=(E<=E_sorted[min(maxN,len(E_sorted))-1])
        expander.trunc(U=U.tocoo(),block_marker=bm,kpmask=kpmask)
        if i!=nsite-1:
            expander.H*=scaling_factors[i+1]/scaling_factors[i]
        if show_spec:
            plot_spectrum(E_sorted[:20]-E_sorted[0],offset=[i,0.],lw=1)
            gcf().canvas.draw()
            pause(0.01)
        elist.append(E/scaling_factors[i])
        bms.append(bm)
    return expander.evolutor,bms,elist

class NRGEngine(object):
    '''
    An Engine for Numerical Renormalization Group Theory.
    '''
    def __init__(self,threadz=False):
        self.threadz=threadz

    def run(self,chain):
        '''
        Run renormalization process.

        chain:
            the mapped 1D chain.
        '''
        chains=[chain] if isinstance(chain,ScaledChain) else chain
        chain0=chains[0]
        Ns=chain0.scale.N
        gatherH=len(self.mops)>0

        for i in xrange(-1,Ns):
            print '\nCalculating H(i) for %s-th site(-1 for impurity).'%(i,)
            for ch in chains:
                ch.scale.set_pinpoint(i)
                ch.HNquickexpand(N=i,threadz=self.threadz,gather=gatherH)
                if ch.ghost!=None:
                    ch.ghost.scale.set_pinpoint(i)
                    ch.ghost.HNquickexpand(N=i,threadz=self.threadz,gather=gatherH)
                if (RANK==0 or not self.threadz):
                    self.do_measure(ch)
                    print 'Matrix Dimension - ',ch.hdim

        #save datas
        if RANK==0 or not self.threadz:
            self.figmanager.saveall()
            for ch in chains:
                for op in ch.cache['mops'].values():
                    op.save_flow()
        return chains
