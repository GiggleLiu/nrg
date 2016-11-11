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

ZERO_REF=1e-12

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

    Attributes:
        :bins: 1d array, the bins to assimulate Energies and weights.
        :weights: complex128, the weight in each bin.
    '''
    def __init__(self,bins,weights=None):
        self.bins=asarray(bins)
        if weights is None: weights=zeros(self.nbin,dtype=complex128)
        self.weights=asarray(weights)

    def __str__(self):
        return '''<Binner>, bin -> %s, filling rate -> %s.'''%(len(self.bins),self.nnz*1./self.nbin)

    @property
    def nbin(self):
        '''number of bins.'''
        return len(self.bins)

    @property
    def nnz(self):
        '''the non-empty bins.'''
        return (abs(self.weights)>ZERO_REF).sum()

    def push(self,datas,wl=1):
        '''
        push a set of delta peaks.

        Parameters:
            :datas: 1d array, datas.
            :wl: 1d array, weights.
        '''
        inds=find_closest(self.bins,datas)
        add.at(self.weights,inds,wl)

    def get_spec(self,wlist,smearing_method='gaussian',window=(-Inf,Inf),b=1.):
        '''
        get the spectrum in wlist.

        Parameters:
            :wlist: 1D array, the target w-space.
            :smearing_method: str/list, the smearing method.

                * 'lorenzian'
                * 'gaussian'
                * 'log-gaussian'
                * [(method,slice,bfactor)]
            :window: len-2 tuple, the slice of spectrum, in the window of binned values.
            :b: float, the broadening parameter.
        '''
        bandwidth=self.bins[-1]-self.bins[0]
        N=self.nbin
        nzmask=abs(self.weights)>ZERO_REF
        el=self.bins[nzmask]
        wl=self.weights[nzmask]
        mask2=(el>=window[0])&(el<=window[1])
        wl,el=wl[mask2],el[mask2]
        if smearing_method=='gaussian':
            b=bandwidth/N*b
            alist=array([gaussian(x=w,mean=el,weights=wl,b=b).sum() for w in wlist])
        elif smearing_method=='log-gaussian':
            b=15.*bandwidth/N*b
            alist=log_gaussian_fast(x=wlist,mean=el,weights=wl,b=b*ones(len(wlist)))
        elif smearing_method=='lorenzian':
            pmask=el>0
            obins=concatenate([el[~pmask],[0],el[pmask]])
            b=diff(obins)*1.5*b
            alist=array([lorenzian(x=w,mean=el,weights=wl,b=b).sum() for i,w in enumerate(wlist)])
        else:
            raise Exception('Unknown smearing_method %s'%smearing_method)
        return alist

def get_binner(D,N,scale_type='log',w0=1e-6):
    '''
    get a binner of specific type.

    Parameters:
        :D: len-2 list/float, the band range.
        :N: int, the number of bins.
        :scale_type: str, the type of bins.

            * 'log'
            * 'linear'
            * 'mixed', fill the gap of 'log' ticks near 0 by linear ticks.
        :w0: float, the minimum energy scale for bins.
    '''
    if ndim(D)==0: D=[-D,D]
    wmin,wmax=D
    assert(wmin<0 and wmax>0)
    if tp=='linear':
        bins=linspace(wmin,wmax,N)
        return Binner(bins)
    elif tp=='log':
        bins_pos=logspace(log(w0)/log(10),log(wmax)/log(10),N/2)
        bins_neg=logspace(log(w0)/log(10),log(-wmin)/log(10),N/2)
        bins=concatenate([-bins_neg[::-1],bins_pos])
        return Binner(bins)
    elif tp=='mixed':
        bins_pos=logspace(log(w0)/log(10),log(wmax)/log(10),N/2)
        bins_neg=logspace(log(w0)/log(10),log(-wmin)/log(10),N/2)
        bins_neg_s=arange(0,w0,exp(log(w0)+log(-wmin/w0)/N)-w0)[1:]
        bins_pos_s=arange(0,w0,exp(log(w0)+log(wmax/w0)/N)-w0)
        bins=concatenate([-bins_neg[::-1],-bins_neg_s[::-1],bins_pos_s,bins_pos])
        return Binner(bins)
    else:
        raise Exception('Unknown bin type!')
