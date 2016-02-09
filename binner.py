#/usr/bin/python
'''
Bin method for spectrum calculation.
'''

from numpy import *
from smearlib import log_gaussian_fast,lorenzian,gaussian,log_gaussian_var
from matplotlib.pyplot import *
from matplotlib.collections import LineCollection
import time,pdb

__all__=['Binner']

def find_closest(A,target):
    '''
    Find closest element positions.

    A:
        the array, it should be sorted.
    target:
        the target to find.
    '''
    idx = A.searchsorted(target)
    idx = np.clip(idx,1,len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

class Binner(object):
    '''
    Bin for spectrum calculation.

    bins:
        the bins to assimulate Energies and weights.
    tp:
        the type of
    '''
    def __init__(self,bins,tp='log',w0=None):
        assert(tp=='linear' or tp=='mixed' or tp=='log')
        if tp=='mixed' and w0 is None:
            raise Exception('w0 is required by binner of type `mixed`!')
        self.tp=tp
        self.bins=bins
        self.weights=zeros(self.N,dtype=complex128)
        self.w0=w0

    def __str__(self):
        return '''<Binner-%s>, bin -> %s, occupacy -> %s.'''%(self.tp,len(self.bins),self.nnz*1./self.N)

    @property
    def N(self):
        '''number of bins.'''
        return len(self.bins)

    @property
    def nnz(self):
        '''the non-empty bins.'''
        return (abs(self.weights)>1e-16).sum()

    def push(self,el,wl):
        '''
        push a set of delta peaks.

        el:
            the energy list.
        wl:
            a list of w.
        '''
        inds=find_closest(self.bins,el)
        add.at(self.weights,inds,wl)
        #weights=bincount(inds,wl,minlength=self.N)
        #self.weights+=weights

    def show(self,lw=1,**kwargs):
        '''
        show datas.

        Parameters:
            :lw: number, the line width.
            **kwargs, key word arguments for LineCollection.
        '''
        bins=self.bins
        ax=gca()
        lc=LineCollection([[(bins[i],0),(bins[i],self.weights[i])] for i in xrange(self.N)],**kwargs)
        lc.set_linewidth(lw)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
 
    def get_spec(self,wlist,smearing_method,b=1.,b0=None):
        '''
        get the spectrum in wlist.

        Parameters:
            :wlist: 1D array, the target w-space.
            :smearing_method: str, the smearing method.
            :b/b0: float, the broadening(b0 for linear region in mixed mode).
        '''
        bandwidth=self.bins[-1]-self.bins[0]
        btp=self.tp
        N=self.N
        nzmask=abs(self.weights)>1e-15
        el=self.bins[nzmask]
        wl=self.weights[nzmask]
        if smearing_method=='gaussian':
            if btp=='linear':
                b=bandwidth/N*b
                alist=array([gaussian(x=w,mean=el,weights=wl,b=b).sum() for w in wlist])
            elif btp=='log':
                b=15.*bandwidth/N*b
                alist=log_gaussian_fast(x=wlist,mean=el,weights=wl,b=b*ones(len(wlist)))
            elif btp=='mixed':
                b=15.*bandwidth/N*b
                alist=log_gaussian_var(x=wlist,mean=el,weights=wl,b=b,w0=self.w0,b0=b0)
        elif smearing_method=='lorenzian':
            if btp=='linear':
                b=bandwidth*1.5*b/N
                alist=array([lorenzian(x=w,mean=el,weights=wl,b=b).sum() for w in wlist])
            elif btp=='log' or btp=='mixed':   #not recommended!
                pmask=el>0
                obins=concatenate([el[~pmask],[0],el[pmask]])
                b=diff(obins)*1.5*b
                alist=array([lorenzian(x=w,mean=el,weights=wl,b=b).sum() for i,w in enumerate(wlist)])
        else:
            raise Exception('Unknown smearing_method for binner type %s'%self.tp)
        return alist

def get_binner(D,N,tp='log',w0=1e-6):
    '''
    get a binner of specific type.

    D:
        the band range.
    N:
        the number of bins.
    tp:
        the type of bins.
        `log` -> logarithmic bins.
        `linear` -> linear bins.
    w0:
        the minimum energy scale for bins.
    '''
    if ndim(D)==0: D=[-D,D]
    wmin,wmax=D
    assert(wmin<0 and wmax>0)
    if tp=='linear':
        bins=linspace(wmin,wmax,N)
        return Binner(bins,tp=tp)
    elif tp=='log':
        bins_pos=logspace(log(w0)/log(10),log(wmax)/log(10),N/2)
        bins_neg=logspace(log(w0)/log(10),log(-wmin)/log(10),N/2)
        bins=concatenate([-bins_neg[::-1],bins_pos])
        return Binner(bins,tp=tp)
    elif tp=='mixed':
        bins_pos=logspace(log(w0)/log(10),log(wmax)/log(10),N/2)
        bins_neg=logspace(log(w0)/log(10),log(-wmin)/log(10),N/2)
        bins_neg_s=arange(0,w0,exp(log(w0)+log(-wmin/w0)/N)-w0)[1:]
        bins_pos_s=arange(0,w0,exp(log(w0)+log(wmax/w0)/N)-w0)
        bins=concatenate([-bins_neg[::-1],-bins_neg_s[::-1],bins_pos_s,bins_pos])
        return Binner(bins,tp=tp,w0=w0)
    else:
        raise Exception('Unknown bin type!')
