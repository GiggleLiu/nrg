from scipy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import scipy.sparse as sps
import pdb,time,copy

from tba.hgen import H2G,SuperSpaceConfig,op_from_mats,op_on_bond
from tba.lattice import Bond
from nrgmap import construct_tridmat
from rglib.mps import op2collection

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

__all__=['ScaledChain','construct_chainopc']

def construct_chainopc(impurity,chain,scale):
    '''
    Construct <OpCollection> for a chain.

    Parameters:
        :impurity: <RGImp>, an impurity instance.
        :chain: <Chain>, the chain instance.
        :scale: <EScale>, the scales.

    Return:
        <OpCollection>, the operator collection.
    '''
    opcs=[]
    nsite=chain.nsite+1
    spaceconfig1=impurity.spaceconfig
    config=list(spaceconfig1.config)
    config[-2]=nsite
    spaceconfig=SuperSpaceConfig(config)
    H0=impurity.H0
    for iz in xrange(chain.nz):
        elist=concatenate([[H0],chain.elist[:,iz]],axis=0)
        tlist=concatenate([chain.t0,chain.tlist[:,iz]],axis=0)
        if chain.is_scalar:
            tlist2=tlist.conj()
        else:
            tlist2=swapaxes(tlist.conj(),-1,-2)
        ebonds=[Bond(zeros(2),i,i) for i in xrange(len(elist))]
        tbonds=[Bond([1.,0],i,i+1) for i in xrange(len(tlist))]
        tbonds2=[Bond([-1.,0],i+1,i) for i in xrange(len(tlist))]
        opc=op_on_bond('E',spaceconfig=spaceconfig,mats=elist,bonds=ebonds)
        opc=opc+op_on_bond('T',spaceconfig=spaceconfig,mats=concatenate([tlist2,tlist],axis=0),bonds=tbonds2+tbonds)
        opc.label='H'
        opcs.append(opc)
    pdb.set_trace()
    return opcs

class ScaledChain(object):
    '''
    A Chain rescaled by 'NRG energy scale'

    chain:
        the original chain.
    scale:
        the scale where chain is defined.
    impurity:
        the impurity.
    H0:
        a list of RGHamiltonian instance of impurity site and it's coupling site.
    label:
        the label of this chain, default is 'Chain'.
    '''
    def __init__(self,chain,scale,impurity,H0,label='Chain'):
        self.label=label
        self.impurity=impurity
        tlist=concatenate([chain.t0[newaxis,...],chain.tlist],axis=0)
        #set scaling factor, the scaling factor here is naive.
        scale.scaling_factor=scale.Lambda**(arange(len(chain.elist))/2.-0.5)
        scale.scaling_factor_relative=scale.scaling_factor/append([1],scale.scaling_factor[:-1])
        if ndim(tlist)<=2:
            self.tlist=tlist*scale.scaling_factor[:,newaxis]
            self.elist=concatenate([[impurity.H0],chain.elist*scale.scaling_factor[:,newaxis]],axis=0)
        else:
            self.tlist=tlist*scale.scaling_factor[:,newaxis,newaxis,newaxis]
            self.elist=concatenate([[[impurity.H0]*scale.nz],chain.elist*scale.scaling_factor[:,newaxis,newaxis,newaxis]],axis=0)
        self.scale=scale

        self.HN=[deepcopy(H0) for i in xrange(scale.nz)]

        #TD-NRG
        self.time=0
        self.ghost=None

    @property
    def nband(self):
        '''return true if spin up and down coupled to different band.'''
        if self.tlist.shape[-1]>2:
            return self.tlist.shape[-1]
        else:
            return 1

    @property
    def hdim(self):
        '''
        Get the current hamiltonian dimension.
        '''
        return [HN.ndim for HN in self.HN]

    @property
    def N(self):
        '''the length of the chain.'''
        return self.scale.N

    @property
    def tlist_original(self):
        '''the hopping terms before scaling.'''
        if ndim(self.tlist)<=2:
            tlist=self.tlist/self.scale.scaling_factor[:,newaxis]
        else:
            tlist=self.tlist/self.scale.scaling_factor[:,newaxis,newaxis,newaxis]
        return tlist

    @property
    def elist_original(self):
        '''the on-site energies before scaling.'''
        if ndim(self.elist)<=2:
            elist=self.elist/self.scale.scaling_factor[:,newaxis]
        else:
            elist=self.elist/self.scale.scaling_factor[:,newaxis,newaxis,newaxis]
        return elist

    def create_ghost(self):
        '''create a ghost.'''
        ghost=deepcopy(self)
        ghost.impurity=None
        self.ghost=ghost

    def show_params(self,logscale=False,*args,**kwargs):
        '''
        Plot parameters.

        logscale:
            logscale for parameter tn, default False
        '''
        tlist=self.tlist
        if logscale:
            tlist=log(tlist)
        xlist=arange(self.scale.N)
        plot(xlist,self.__elist__,*args,**kwargs)
        plot(xlist,tlist,*args,**kwargs)
        legend(['e0','t'])

    def get_opstring(self,i):
        '''
        Get the relevent op strings.
        
        Parameters:
            :i: the site index.

        Return:
            list, a list of <OpString>/<OpUnit>
        '''
        e=self.elist[i]
        t=self.tlist[n+1]

    def HNexpand(self,**kwargs):
        '''
        expand Hamiltonian.

        **kwargs:
            refer specific expansion Hamiltonian.
        '''
        rescalefactor=self.scale.get_rescalefactor()
        for i in xrange(self.scale.nz):
            params={}
            for param in kwargs:
                value=kwargs[param]
                if hasattr(value,'__iter__'):
                    params[param]=value[i]
                else:
                    params[param]=value
            self.HN[i].expand(**params)
            self.HN[i].rescalefactor=rescalefactor

    def HNquickexpand(self,N,threadz=False,gather=False,**kwargs):
        '''
        expand HN, the quick version.

        N:
            the iteration depth 
        threadz/gather:
            multi-threading over z if True/gather after threading if True..
        **kwargs:
            the parameters.
        '''
        rescalefactor=1. if N<0 else self.scale.scaling_factor[N]
        islast=(N==self.N-1)
        H=[]
        #prepair parameters
        if N==-1:
            impurity=self.impurity
            if impurity is None:
                return
            elif impurity.tp=='Anderson':
                #cparams=[{'U':impurity.U,'e0':impurity.ed,'Bz':impurity.Bz}]
                cparams=[{'U':impurity.U,'e0':repeat(impurity.H0[newaxis,...],self.scale.nz,axis=0)}]   #pay attention to the nz axis!
            elif impurity.tp=='Kondo':
                cparams=[{},{'J':impurity.J,'e0':impurity.ed}]
            else:
                raise Exception('Unrecognized impurity!@ScaledChain.HNquickexpand')
        else:
            cparams=[{'factor':self.scale.scaling_factor_relative[N],'t':self.t(N-1),'e0':self.e0(N)}]

        #start to expand
        ntask=(self.scale.nz-1)/SIZE+1
        for i in xrange(self.scale.nz):
            if i/ntask==RANK or not threadz:
                #prepair parameters
                for cparam in cparams:
                    params={}
                    for param in cparam:
                        value=cparam[param]
                        if ndim(value)==0:
                            params[param]=value
                        else:
                            params[param]=value[i]  #multiple z
                    self.HN[i].rescalefactor=1./rescalefactor
                    self.HN[i].expand(ops)
                H.append(self.HN[i])

        if SIZE>1 and threadz:
            if gather:
                gather_H(H,self.HN,COMM,SIZE,RANK)
        else:
            self.HN=H

    def HNtruncate(self):
        '''truncate hamiltonian and operators.'''
        for i in xrange(self.scale.nz):
            self.HN[i].trunc()
        if self.ghost is not None:
            self.ghost.HNtruncate()

    def get_G0(self,wlist,Gap=0.,geta=1e-2):
        '''
        get the Green's function without interaction by method of recursive Green's function.

        wlist:
            a list of frequency.
        '''
        Lambda=self.scale.Lambda
        tlist=self.tlist_original
        elist=self.elist_original
        e0=self.impurity.H0
        nz=elist.shape[1]

        ion()
        print 'Generating Green\'s function ...'
        dlv=0;dle=0
        G0l=[]
        for iz in xrange(nz):
            print 'Running for %s-th z number.'%iz
            el=elist[:,iz]
            tl=tlist[:,iz]
            dl=[]
            gl=[]
            for w in wlist:
                sigma=0
                for e,t in zip(el[::-1],tl[::-1]):
                    g0=H2G(w=w,h=e+sigma,geta=geta)
                    tH=transpose(conj(t))
                    sigma=dot(tH,dot(g0,t))
                gl.append(H2G(w=w,h=e0+sigma,geta=geta))
            G0l.append(gl)
        return mean(G0l,axis=0)

    @property
    def H0(self):
        '''
        A list of hamiltonian without interaction.
        '''
        tlist=self.tlist_original
        elist=self.elist_original
        e0=self.impurity.H0
        nz=elist.shape[1]
        HL=[]
        for iz in xrange(nz):
            tl=tlist[:,iz]
            tlH=swapaxes(tl,1,2).conj()
            el=concatenate([e0[newaxis,...],elist[:,iz]],axis=0)
            offset=[-1,0,1]
            data=[tlH,el,tl]
            N=len(data[1])
            B=ndarray([N,N],dtype='O')
            #fill datas
            for i in xrange(N):
                for j in xrange(N):
                    for k in xrange(3):
                        if i-j==offset[k]:
                            B[i,j]=complex128(data[offset[k]+1][min(i,j)])
            HL.append(sps.bmat(B).toarray())
        return HL
