'''
Scale class for NRG.
'''

from scipy import *
from scipy.linalg import norm
from matplotlib.pyplot import *

from utils import text_plot
import pickle

__all__=['EScale','ticker2scale','load_scale']

class EScale(object):
    '''
    Energy Scale for <NRG> utilities.

    Parameter:
        :scale_ticks: len-2 list of 2D array, (data_neg, data_pos) with data_neg=[e(-1),e(-2),...] and data_pos=[e(1),e(2),...].
        :Lambda: float, the logarithmic span.
        :z: float, twisting parameter.
    '''
    def __init__(self,scale_ticks,Lambda,z):
        self.Lambda=Lambda
        self.z=z
        #datas
        self._data_neg,self._data=scale_ticks

    def __getitem__(self,n):
        '''
        Get energy scale of specific index, e.g. s[1] = D, is the first index.
        '''
        assert(abs(n)>0 and abs(n)<=self.N+1)
        if n>0:
            return self._data[n-1]
        else:
            return -self._data_neg[-n-1]

    @property
    def N(self):
        '''Number of intervals.'''
        return len(self._data)-1

    @property
    def D(self):
        '''The band-width, [left edge, right edge]'''
        return -self._data_neg[0],self._data[0]

    @property
    def indices(self):
        '''Get the `true` indices of this scale.'''
        z=self.z
        return append(arange(1+z,self.N+1+z),arange(-z-1,-self.N-z-1,-1))

    def get_scaling_factor(self,i):
        '''
        Get the scaling factor for site(interval) i.

        Parameters:
            :i: integer, the index of interval.

        Return:
            float, the scaling factor.
        '''
        return self.Lambda**(i/2.)

    def show(self,hspace=1.,offset=(0,0),**kwargs):
        '''
        show scale.
        '''
        scale_visual=concatenate([-self._data_neg,self._data[::-1]])+offset[0]
        yi=offset[1]
        plot([scale_visual[0],scale_visual[-1]],[yi,yi],'k',lw=1)
        #scatter(scale_visual[:,i],ones(self.N*2+2)*yi,s=30,color='k',**kwargs)
        plot(scale_visual,ones(self.N*2+2)*yi,marker='$\\bf |$',**kwargs)

    def interval(self,n,N=2):
        '''
        Get a span of scale, e.g. 

            * the first interval of positive branch s.interval(1)
            * the first interval of negative branch s.interval(-1)

        Parameters:
            :n: integer, the index of energy scale 1,2,3 ... or -1,-2,-3 ...
            :N: integer, the number of samples, default is 2 - [vmin,vmax].

        Return:
            1D array, a linear space in this interval.
        '''
        if n==0 or n>self.N:
            raise ValueError('index out of range.')
        sgn=-1 if n<0 else 1
        n=abs(n)-1
        if n>=self.N:
            raise Exception('Error','Interval index exceeded, it should be smaller than %s but got %s'%(self.N,n))
        if sgn<0:
            vmin=self._data_neg[n]
            if n!=self.N-1:
                vmax=self._data_neg[n+1]
            else:
                vmax=vmin/float64(self.Lambda)
            return -linspace(vmin,vmax,N)
        else:
            vmax=self._data[n]
            if n!=self.N-1:
                vmin=self._data[n+1]
            else:
                vmin=vmax/float64(self.Lambda)
            return linspace(vmin,vmax,N)

    def save(self,token):
        '''
        save scale.

        Parameters:
            :token: the token for filename.
        '''
        filename=token+'.dat'
        f=open(filename,'w')
        pickle.dump(self,f)
        f.close()

def load_scale(token):
    '''
    load scale.

    Parameters:
        :token: the token for filename.
    '''
    filename=token+'.dat'
    f=open(filename,'r')
    scale=pickle.load(f)
    f.close()
    return scale


def ticker2scale(tickers,N,z):
    '''
    Get <EScale> from <Ticker>s.

    Parameters:
        :tickers: len-2 list of <Ticker>/tick function.
        :N: integer, the length of scale.
        :z: float, twisting parameter.

    Return:
        <EScale>, the scale instance.
    '''
    Lambda=tickers[0].Lambda
    scale_ticks=[tickers[0](arange(1,N+2)+z),tickers[1](arange(1,N+2)+z)]
    scale=EScale(scale_ticks,Lambda,z)
    return scale
