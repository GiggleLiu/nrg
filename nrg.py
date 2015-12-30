from scipy import *
from matplotlib.pyplot import *
from setting.local import COMM,SIZE,RANK
from tdnrg import RGTimeLine
from core.phcore import FigManager
from rgobj import RGobj
from argobj import ARGobj
from scaledchain import ScaledChain
import pdb,time

class NRGEngine(object):
    '''
    An Engine for Numerical Renormalization Group Theory.
    '''
    def __init__(self,threadz=False):
        self.mops=[]
        self.mops_a=[]
        self.figmanager=FigManager()
        self.chaindict={}
        self.chain=None
        self.threadz=threadz
        self.setting={
                }

    @property
    def trackrho(self):
        '''return True if need to track rho.'''
        return len(self.mops_a)>0

    def __cope_requirements__(self,chain):
        '''
        cope with requirements.

        chain:
            the chain requirements are for.
        '''
        if self.trackrho:
            chain.HNtrackrho(True)
        for op in self.mops+self.mops_a:
            target_chain=op.setting['target_chain']
            if target_chain==None or any(target_chain==chain.label):
                if isinstance(op,RGobj):
                    chain.register_op(op)
                if op.subtract_env and chain.ghost==None:
                    chain.create_ghost()
                for rq in op.requirements:
                    if rq.tp=='op':
                        chain.HNregistercovop(rq.info['kernel'])
                        if op.subtract_env:
                            chain.ghost.HNregistercovop(rq.info['kernel'])
                    if rq.islist:
                        chain.HNtrack(rq)
    
    def register_measure(self,op,target_chain=None,graphic=True):
        '''
        Register Operators for measurements.
        '''
        op.setting['target_chain']=target_chain
        if isinstance(op,RGobj):
            self.mops.append(op)
            if graphic and (RANK==0 or not self.threadz):
                fig,ax,pls=self.figmanager.register(op.label,npls=op.datalen)
                ax.set_xlabel('Temperature')
                ax.set_ylabel(op.label)
        elif isinstance(op,ARGobj):
            self.mops_a.append(op)
        else:
            raise Exception('Error','Measurable objects not qualified!')

    def register_chain(self,chain):
        '''
        multiple chain support.
        '''
        self.chains.append(chain)


    def do_measure(self,chain):
        '''
        Measure An operator.
        
        chain:
            the chain to measure.
        '''
        scale=chain.scale
        N=scale.pinpoint
        beta=scale.get_beta()
        Ntick=5
        mops=chain.cache['mops']
        HN=chain.HN
        ghost_on=chain.ghost!=None

        for opname in mops:
            op=mops[opname]
            #if N is not a measure point of op, skip.
            if not op.measurepoint(N):
                continue
            #mesure
            mval=[]
            for i in xrange(scale.nz):
                cmval=op.get_expect(HN[i],beta=beta)
                if op.subtract_env:
                    cmval-=chain.ghost.cache['mops'][opname].get_expect(chain.ghost.HN[i],beta=beta)
                mval.append(cmval)

            #update values of operator(the real chain but not the ghost one).
            op.update(mval)
            #display it
            temperature=append([1.],1./chain.scale.scaling_factor[:N+1])/0.75
            op.show_flow(graph=self.figmanager[op.label],xdata=temperature)

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

        #initialize Hamiltonians and operators of chain
        for ichain,ch in enumerate(chains):
            print '\nCoping requirements for chain-%s.'%(ch,)
            self.__cope_requirements__(ch)

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
