from numpy import *
from fysics import *

__all__=['gaussian','log_gaussian','log_gaussian_var','log_gaussian_fast','lorenzian']

def gaussian(x,mean,b,weights=1.):
    '''
    Gaussian distribution.

    Parameters:
        :x: ndarray, x or a list of x.
        :mean: num, mean value.
        :b: num, broadening.
        :weights: ndarray, weights of specific spectrum, it should take the same shape as mean.

    Return: ndarray,
    '''
    return 1./b/sqrt(pi)*exp(-((x-mean)/b)**2)*weights

def log_gaussian(x,mean,weights,b=1.):
    '''
    Logarithmic Gaussian broadening for peak.

    NOTE:
        it is asymmetric about mean and -mean.

    Parameters:
        :x: ndarray, x or a list of x.
        :mean: num, mean value.
        :b: num, broadening.
        :weights: ndarray, weights of specific spectrum, it should take the same shape as mean.

    Return: ndarray,
    '''
    assert(ndim(x)==1 and ndim(mean)==1 and ndim(weights)==1)
    xmask=x>0
    mmask=mean>0
    al=[]
    for xm,mm in [(~xmask,~mmask),(xmask,mmask)]:
        cmean=mean[mm]
        abscmean=abs(cmean)+1e-18
        if any(mm):
            al.append((weights[mm]*exp(-b**2/4.)/b/abscmean/sqrt(pi)*exp(-(log(abs(x[xm,newaxis])/abscmean)/b)**2)).sum(axis=-1))
        else:
            al.append(zeros(xm.sum(),dtype=complex128))
    return concatenate(al)

def log_gaussian_fast(x,mean,weights,b):
    '''
    Logarithmic Gaussian broadening for peak, the fortran version.

    NOTE:
        it is asymmetrix about mean and -mean.

    Parameters:
        :x: ndarray, x or a list of x.
        :mean: num, mean value.
        :b: num, broadening.
        :weights: ndarray, weights of specific spectrum, it should take the same shape as mean.

    Return: ndarray,
    '''
    assert(ndim(x)==1 and ndim(mean)==1 and shape(weights)==shape(mean) and shape(b)==shape(x))
    if len(mean)==0:
        return zeros(x.shape,dtype=weights.dtype)
    xmask=x>0
    mmask=mean>0
    al=[]
    for xm,mm in [(~xmask,~mmask),(xmask,mmask)]:
        if any(mm):
            al.append(flog_gaussian(wlist=x[xm],elist=mean[mm],weights=weights[mm],b=b[xm]))
        else:
            al.append(zeros(xm.sum(),dtype=complex128))
    return concatenate(al)


def log_gaussian_var(x,mean,weights,b,w0,b0=None):
    '''
    Logarithmic Gaussian broadening for peak, the varied fortran version.

    NOTE:
        it is asymmetric about x and -x.

    Parameters:
        :x: ndarray, x or a list of x.
        :mean: num, mean value.
        :b: num, broadening.
        :weights: ndarray, weights of specific spectrum, it should take the same shape as mean.
        :w0/b0: num, the transition points and it's smearing parameter, will use w0 by default.

    Return: ndarray,
    '''
    assert(ndim(x)==1 and ndim(mean)==1 and shape(weights)==shape(mean) and ndim(b)==0)
    if len(mean)==0:
        return zeros(x.shape,dtype=weights.dtype)
    return flog_gaussian_var(wlist=x,elist=mean,weights=weights,b=b,w0=w0,b0=-1 if b0 is None else b0)

def lorenzian(x,mean,b,weights=1.):
    '''
    Lorenzian broadening for a peak.

    Parameters:
        :x: ndarray, x or a list of x.
        :mean: num, mean value.
        :b: num, broadening.
        :weights: ndarray, weights of specific spectrum, it should take the same shape as mean.

    Return: ndarray,
    '''
    return weights*(1./pi/(x-1j*b-mean)).imag


