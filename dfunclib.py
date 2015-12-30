#/usr/bin/python
from numpy import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.linalg import eigvalsh
from matplotlib.pyplot import *
import pdb

from tba.hgen.utils import sx,sy,sz,s2vec

def get_dfunc_pseudogap(Gamma,r,Gap=None,D=1.,dimension=1):
    '''
    D(w) for pseudo-gap system.

    Gamma:
        the overall dos.
    r:
        the r-index.
    Gap:
        the gap value, default D.
    D:
        the band width.
    dimension:
        the dimension.
    '''
    if ndim(D)==0:
        D1=-D;D2=D
    else:
        D1,D2=D
    Gamma=Gamma if dimension==1 else Gamma*identity(dimension)
    if Gap==None: Gap=D
    zero=0. if dimension==1 else zeros([dimension,dimension])
    rho0=Gamma/Gap**r
    if r<0:
        raise warnings.warn('r-index smaller than 0 will cause divergence is hybridization function!')
    def dfunc(w):
        absw=abs(w)
        if absw>D:
            return zero
        elif absw>Gap:
            return Gamma
        else:
            return rho0*absw**r
    return dfunc

def get_dfunc_flat(Gamma,D=1.,dimension=1):
    '''
    get a flat D(w).

    Gamma:
        the overall dos.
    D:
        the band width.
    dimension:
        the dimension.
    '''
    return get_dfunc_pseudogap(Gamma=Gamma,r=0,D=D,dimension=dimension)

def get_dfunc_skewsc(Gap,Gamma,skew,D=1.,eta=1e-10,g=False):
    '''
    get the skewed dfunc for superconductor.

    Gap:
        the gap value.
    Gamma/skew:
        the overall strength, skew of hybridization function.
    D:
        the band-width.
    eta:
        smearing factor, None for matsubara Green's function.
    g:
        get self energy instead of hybridization function.
    '''
    one=identity(2)
    N0=Gamma/pi
    def gfunc(w):
        z=(w+1j*eta) if eta!=None else 1j*w
        sqc=sqrt(Gap**2-z**2)
        I0=-2*arctan(D/(sqc))/sqc
        I2=-2*D+2*sqc*arctan(D/sqc)
        return N0*(I0*z*one-I0*Gap*sx+skew*I2*sz)
    def dfunc(w):
        g=gfunc(w)
        return 1j/2.*(g-g.conj().T)
    if g:
        return gfunc
    else:
        return dfunc

def get_dfunc_sc(Gap,Gamma,D=1.,mu=0.,eta=1e-10,g=False,wideband=False):
    '''
    D(w) for superconducting surface.

    Gap:
        the Gap value.
    Gamma:
        the overall dos.
    D:
        the band width.
    eta:
        the smearing factor.
    g:
        get the green's function(get dfunc by default).
    '''
    if ndim(D)==0:
        D=[-D,D]
    def gfunc_finite(w):
        z=(w+1j*eta)
        sgn=1 if w>0 else -1
        sqc=sqrt(Gap**2-mu**2-z**2)
        res=-Gamma/pi*(((arctan(D[1]/sqc)-arctan(D[0]/sqc))/sqc)*(z*identity(2)-Gap*sx)-0.5*sgn*(log((D[0]**2-z**2+Gap**2)/(D[1]**2-z**2+Gap**2)))*sz)
        return res

    def gfunc(w):
        E=sqrt(Gap**2-mu**2-(w+1j*eta)**2)
        if w<=D[1] or w>=D[0]:
            res=-Gamma*(((w+1j*eta)/E)*identity(2)-(Gap/E)*sx)
        else:
            res=zeros([2,2])
        return res

    if wideband:
        gf=gfunc
    else:
        gf=gfunc_finite

    def dfunc(w):
        g=gf(w)
        return 1j/2.*(g-g.conj().T)

    if g:
        return gf
    else:
        return dfunc

def get_dfunc_scl(Gap,D0=1.,geta=3e-2):
    '''
    D(w) for superconducting material - the lattice model.
    '''
    hkgen=SimpleHkGen(dimension=2,mu=0.,t1=D0,rashba=0.,Delta_s=Gap,Delta_p=0.,Delta_d=0.,Nx=20,SC='snambu')
    hkmesh=hkgen.gethkmesh()
    def dfunc(w):
        gkmesh=hkmesh.getgmesh(w,tp='r',geta=geta)
        g0=sum(gkmesh,axis=tuple(arange(hkgen.dimension)))/prod(gkmesh.shape[:2])
        return -g0.imag
    return dfunc

def get_dfunc_dwave(Gap,Gamma,D=1.,which='anti-nodal',t1=0.,t2=0.,N=2000,dimension=2):
    '''
    Get the hybridization function for an idea d-wave surface.

    Gap:
        the gap value.
    Gamma:
        DOS near the fermi surface.
    D:
        the band-width
    which: 
        'n' -the nodal direction. 
        'an'-the anti-nodal  case.
    '''
    if dimension==1 and (t1**2+2*t2)!=0:
        raise Exception('Error','Can not get 1-D dfunc for model of extended dwave superconductor.')
    #wlist=append(linspace(0,Gap*2,N/4),linspace(Gap*2,D,N/4)[1:])
    #wlist=append(-wlist[:1:-1],wlist)
    wlist=logspace(-16,0,N/2)
    wlist=append(-wlist[::-1],wlist)
    d0l=[]
    d1l=[]
    for w in wlist:
        if w==0:
            #avoid singular point
            d0l.append(0);d1l.append(0)
            continue
        a=0 if abs(w)>Gap else arccos(abs(w)/Gap)/2.
        b=pi/2.-a
        if which=='anti-nodal':
            d0=quad(lambda th:abs(w)/sqrt(abs(w**2-(Gap*cos(2*th))**2) or 1e-18),a=a,b=b,limit=200)
            d1=quad(lambda th:w/abs(w)*min(abs(w),Gap)/sqrt(abs(w**2-(Gap*cos(2*th))**2) or 1e-18),a=a,b=b,limit=200)
        elif which=='nodal':
            d0=quad(lambda th:sqrt(abs(w**2-(Gap*cos(2*th))**2))/w,a=a,b=b)
            d1=0
        d0l.append(d0[0]);d1l.append(d1[0])
    d0l=Gamma*(1+2*t1**2+2*t2**2)*array(d0l)
    d0func=interp1d(wlist,d0l)
    if dimension==1:
        return d0func
    else:
        d1l=Gamma*(t1**2+2*t2)*array(d1l)
        d1func=interp1d(wlist,d1l)
        dfunc=lambda w:d0func(w)*identity(2)+d1func(w)*sx if abs(w)<=D else zeros([2,2])
        #ion()
        #plot(wlist,[eigvalsh(dfunc(w)) for w in wlist])
        #plot(wlist,d0func(wlist))
        #plot(wlist,d1func(wlist))
        #pdb.set_trace()
        return dfunc

def check_dfunc(dfunc,nband,D,Gap=0.,method='eval'):
    '''
    Checking for dfunc.

    dfunc:
        the hybridization function.
    nband:
        the number of bands.
    method:
        the method for checking.
        * `pauli` -> pauli decomposition for 2D.
        * `eval` -> check for eigenvalue.
    '''
    ion()
    if nband!=2 and method=='pauli':
        warnings.warn('Checking pauli components is not allowed for non-2band bath!')
        method='eval'
    if ndim(D)==0:
        D=[-D,D]
    wlist=linspace(D[0],D[1],1000)
    if nband==1:
        dl=array([dfunc(w) for w in wlist])
    elif method=='eval':
        dl=array([eigvalsh(dfunc(w)) for w in wlist])
    elif method=='pauli':
        dl=array([s2vec(dfunc(w)) for w in wlist])
    plot(wlist,dl)
    if method=='eval':
        legend([r'$\rho_%s$'%i for i in xrange(nband)])
    elif method=='pauli':
        legend([r'$\rho_0$',r'$\rho_x$',r'$\rho_y$',r'$\rho_z$'])
    pdb.set_trace()
