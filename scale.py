#!/usr/bin/python
'''
Scale class for NRG.
'''
from scipy import *
from scipy.linalg import norm
from matplotlib.pyplot import *
from utils import text_plot
import pickle

__all__=['EScale']

class EScale(object):
    '''
    Energy Scale for <NRG> utilities.

    Parameter:
        :scale_ticks: len-2 list of 2D array, (data_neg, data_pos) with data_neg=[e(-1),e(-2),...] and data_pos=[e(1),e(2),...].
        :Lambda: float, the logarithmic span.
        :z: float/1D array, twisting parameter, it is used to smear the fluctuation brought by NRG truncation. 
        :pinpoint: integer, the current iteration.
        :tick_type: float/None, the type of tick.
        :scaling_factor: 2D array, the scaling factors.
        :scaling_factor_relative: 2D array, the relatice scaling factor.
    '''
    def __init__(self,scale_ticks,Lambda,z,tick_type=None):
        self.Lambda=Lambda
        if ndim(z)==0:
            z=array([z])
        self.z=z
        if any((z>1) | (z<=0.)):
            raise Exception('Error','Using non-appropriate discretization offset z=%s'%z)
        self.pinpoint=-1   #current scale, it is a 'running' scale now. -1 is the impurity site.
        self.tick_type=tick_type
        #datas
        self.__datan__,self.__data__=scale_ticks
        #scaling factor is in principle, independant of tick positions.
        self.scaling_factor=None
        self.scaling_factor_relative=None

    def __getitem__(self,n):
        '''
        Get energy scale of specific index.
        '''
        if n>0:
            return self.__data__[n]
        else:
            return self.__datan__[-n]

    @property
    def N(self):
        return len(self.__data__)

    @property
    def D(self):
        return -self.__datan__[0,0],self.__data__[0,0]

    @property
    def nz(self):
        '''
        Number of z.
        '''
        return len(self.z)

    @property
    def temperature_list(self):
        '''
        get equivalent list of temperature of scale.
        '''
        return 1./self.beta_list

    @property
    def beta_list(self):
        '''
        get equivalent list of beta of scale.
        '''
        return self.get_beta(arange(self.N))

    @property
    def ticks(self):
        '''
        Get the full scale.
        '''
        #for indexing order ->
        return concatenate([self.__data__,-self.__datan__])

    @property
    def xlist(self):
        '''
        Get the x values of this scale.
        '''
        return array([append(arange(1+z,self.N+1+z),arange(-z-1,-self.N-z-1,-1)) for z in self.z]).T

    def get_beta(self,n=None):
        '''
        Get equivilent beta.

        n:
            the scale index, pinpoint by default.
        '''
        BETA0=0.75
        if n==None: n=self.pinpoint
        sfactor=norm(self.D)/sqrt(2.)/self.scaling_factor[n]
        return BETA0/sfactor

    def d(self,n):
        '''
        Get the length of interval of n-th scale.
        '''
        intv=self.interval(n)
        return intv[:,1]-intv[:,0]

    def show(self,hspace=1.,**kwargs):
        '''
        show scale.
        '''
        scale_visual=concatenate([-self.__datan__,self.__data__[::-1]])
        for i in xrange(self.nz):
            yi=i*hspace
            plot([scale_visual[0],scale_visual[-1]],[yi,yi],'k',lw=1)
            #scatter(scale_visual[:,i],ones(self.N*2)*yi,s=30,color='k',**kwargs)
            plot(scale_visual[:,i],ones(self.N*2)*yi,color='k',marker='$\\bf |$',**kwargs)

    def show_running(self,ticks='n',ax=None):
        '''
        show scale as a 'running chain'.

        ticks:
            `n`: scale index.
            `t`: temperatures.
        ax:
            axis.
        '''
        ax=ax or gca()
        #display a impurity
        ax.scatter(-1,0,marker='s',s=100,c='k')
        #display a chain.
        x,y=arange(self.N),zeros(self.N)
        if ticks=='t':
            tl=array(['%.1e'%t for t in self.temperature_list])
            tw=0.06
        else:
            tl=x
            tw=0.02
        c=concatenate([repeat([[0,0.7,0.7]],(self.pinpoint),axis=0),repeat([[0.7,0.7,0.7]],self.N-self.pinpoint,axis=0)],axis=0)
        ax.scatter(x,y,c=c,s=50)
        text_plot(x_data=x,y_data=y,texts=tl,ax=ax,txt_width=tw)

    def set_pinpoint(self,n):
        '''set pin point.'''
        self.pinpoint=n

    def interval(self,n,N=2):
        '''
        Get a span of scale.

        n:
            the index of energy scale 1,2,3 ... or -1,-2,-3 ...
        N:
            the number of samples, default is 2 - [vmin,vmax].
        '''
        sgn=-1 if n<0 else 1
        n=abs(n)-1
        if n>=self.N:
            raise Exception('Error','Interval index exceeded, it should be smaller than %s but got %s'%(self.N,n))
        if sgn<0:
            vmin=array([float64(self.__datan__[n,i]) for i in xrange(self.nz)])
            if n!=self.N-1:
                vmax=array([float64(self.__datan__[n+1,i]) for i in xrange(self.nz)])
            else:
                vmax=vmin/float64(self.Lambda)
            return -array([linspace(vmin[i],vmax[i],N) for i in xrange(self.nz)])
        else:
            vmax=array([float64(self.__data__[n,i]) for i in xrange(self.nz)])
            if n!=self.N-1:
                vmin=array([float64(self.__data__[n+1,i]) for i in xrange(self.nz)])
            else:
                vmin=vmax/float64(self.Lambda)
            return array([linspace(vmin[i],vmax[i],N) for i in xrange(self.nz)])

def save_scale(token,scale):
    '''
    save scale.

    token:
        the token for filename.
    scale:
        an EScale instance.
    '''
    filename='data/scale_%s_%s_%s_%s.npy'%(token,scale.Lambda,scale.N,scale.nz)
    f=open(filename,'w')
    pickle.dump(scale,f)
    f.close()

def load_scale(token,Lambda,N,nz):
    '''
    load scale.

    token:
        the token for filename.
    '''
    filename='data/scale_%s_%s_%s_%s.npy'%(token,Lambda,N,nz)
    f=open(filename,'r')
    ticks=pickle.load(f)
    f.close()
    return ticks


