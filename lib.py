from scipy import *
from matplotlib.pyplot import *
from scipy import sparse as sps
import pdb,time

from rglib.hexpand import MaskedEvolutor,ExpandGenerator
from blockmatrix import eigbh,trunc_bm
from impuritymodel import ImpurityModel,scale_bath

__all__=['Tscale_qnumber','smear_evenodd']

def Tscale_qnumber(mps,EL,TL,qfunc=None):
    '''
    Get the finite temperature scaling for good quantum number observable.

    Prameters:
        :mps: <MPS>, for RNG.
        :EL: list of 1d array, energy spectrum at each iteration.
        :TL: 1d array, tipical temperature for each iteration.
        :qfunc: function/None, the function on qnumber before evaluation.

    Return:
        2d array, (T, qnum).
    '''
    #get inflated block marker.
    bms=[a.labels[2].bm.inflate() for a in mps.AL]
    res=[]
    for E,bm,T in zip(EL,bms,TL):
        rho=exp(-E/T)
        if qfunc is None:
            q=(rho[:,newaxis]*bm.labels).sum(axis=0)/sum(rho)
        else:
            q=(rho[:,newaxis]*qfunc(bm.labels)).sum(axis=0)/sum(rho)
        res.append(q)
    res=array(res)
    return res

def smear_evenodd(TL,OL):
    '''
    smearing the even odd oscillation by applying
            O(T_N)=0.5*(O(N)+O(N-1)+[O(N+1)-O(N-1)]/(T_N+1 - T_N-1)*(T_N - T_N-1))

    Parameters:
        :TL: 1d array, temperature.
        :OL: ndarray, operator.

    Return:
        nd array, smeared operator.
    '''
    r1,r2,r3=OL[:-2],OL[1:-1],OL[2:]
    TL=TL.reshape([-1]+[1]*(ndim(OL)-1))
    t1,t2,t3=TL[:-2],TL[1:-1],TL[2:]
    res=concatenate([OL[:1],0.5*(r2+r1+(r3-r1)*(t2-t1)/(t3-t1)),OL[-1:]],axis=0)
    return res
