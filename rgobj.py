from scipy import *
from matplotlib.pyplot import *
from scipy.interpolate import splrep,splev
import pdb,time

from tba.hgen import op_M

__all__=['RGobj','RGobj_Tchi']

class RGobj(object):
    '''
    Measuring objects used for NRG.
    '''
    def measure(self,*args,**kwargs):
        raise NotImplementedError()

class RGobj_N(RGobj):
    '''
    occupacy of specific site.

    spaceconfig:
        it should be provided if it runs in 'o' mode.
    spinconfig:
        configuration of spins, default is 'up'.
        refer spindict for more options('dn','upup+dndn'/'s0','upup-dndn'/'sz' ...).
    label:
        the label, by default it is 'chi'.
    site:
        the site index. defalt is 0(the impurity site).
    '''
    def __init__(self,spinconfig='up',label='n',site=0,*args,**kwargs):
        super(RGobj_N,self).__init__(label,*args,**kwargs)
        self.site=site
        n=0
        self.rop=rgop_n(site,spinconfig)
        self.rop.label=self.label
        self.requirements.append(RGRequirement(self.rop.label,tp='op',info={'kernel':self.rop}))

    def get_expect(self,HN,beta,*args,**kwargs):
        '''
        Get the expectation value of N.

        HN:
            the hamiltonian instance.
        beta:
            inverse of temperature.
        '''
        nop=HN.collect_op(self.rop.label)
        E=HN.E_rescaled
        rho=HN.rho(T=1./beta)
        mval=(rho.multiply(nop.T)).sum()
        return mval

class RGobj_Tchi(RGobj):
    '''
    Physics quantity T*chi for spin (non-)conservative systems.
    T*chi=<sz^2>-<sz>^2.

    mode:
        's' -> spin is a good quantum number for hamiltonian.
        'on' -> using subtraction of spin z operators.
    spaceconfig:
        it should be provided if it runs in 'on' mode.
    '''
    def __init__(self,mode='s',spaceconfig=None):
        super(RGobj_Tchi,self).__init__()
        assert(mode in ['on','s'])
        if mode=='on': assert(spaceconfig is not None)
        self.mode=mode
        self.spaceconfig=spaceconfig

        if mode=='on':
            sz=op_M(spaceconfig,direction='z')()/2

    def measure(self,data,elist,scaling_factors,beta_factor=0.4,M_axis=None,*args,**kwargs):
        '''
        Get the expectation value of Tchi.

        Parameters:
            :data: <BlockMarker>/<MPS>, the data to measure.
            :elist: list of length nsite, a list of rescaled energy.
            :scaling_factors: 1D array, the scaling factors.
            :beta_factor: float, the factor for beta with respect to current scaling factor.
            :M_axis: integer, the axis of good quantum number M.
        '''
        if M_axis is None and self.mode=='s':
            raise ValueError('please specify the spin axis of the block marker.')
        res=[]
        nsite=len(elist)
        beta_list=beta_factor*scaling_factors
        for j in xrange(nsite):
            if self.mode=='on':
                siself.__get_expect_o__(mps=data[j],elist=elist[j],beta=beta_list[j],*args,**kwargs)
            else:
                si=self.__get_expect_s__(block_marker=data[j],E_true=elist[j],beta=beta_list[j],M_axis=M_axis,*args,**kwargs)
            res.append(si)
        return array(res)

    def __get_expect_s__(self,block_marker,E_true,beta,M_axis):
        '''
        Get expectation value of sussceptibility chi.
        In the Q,S,m revserved case, we can use the following relation ->
            chi = S(S+1)/3 x (2*S+1) - sum(sz**2 for sz in range(-S,S))

        Parameters:
            :block_marker, M_axis: the block marker and the axis of good quantum number 'M'.
            :E_true: the energy list rescaled back.
            :beta: the inverse temperature.
        '''
        E=E_true-E_true.min()
        Z=sum(exp(-beta*E))
        mval=0.
        szm=0.  #mean of Sz
        labels=block_marker.labels
        for i in xrange(block_marker.nblock):
            Ei=block_marker.extract_block(E,i)
            rho=sum(exp(-beta*Ei))
            #ind is Sz
            sz=labels[i][M_axis]/2.
            mval+=rho*sz**2/Z
            szm+=rho*sz/Z
        mval-=szm**2
        return mval

    def __get_expect_o__(self,HN,beta,*args,**kwargs):
        '''
        Get expectation value of sussceptibility chi.
        Tchi = <m**2>-<m>**2

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        raise NotImplementedError()

class RGobj_ABS(RGobj):
    '''
    Physics quantity of Andreev Bound state Energy:
        it is defined as minimum exitation energy?

    label:
        the label.
    nabs:
        the number of exitation energies.
    '''
    def __init__(self,label='ABS Energy',nabs=2,*args,**kwargs):
        super(RGobj_ABS,self).__init__(label,datalen=nabs,*args,**kwargs)

    def get_expect(self,HN,beta,*args,**kwargs):
        '''
        Get the spectrum of ABS states.

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        rescalefactor:
            the rescaling factor for energy.
        '''
        E=sort(HN.E_true)[:self.datalen+1]
        if len(E)<self.datalen+1:
            print 'GAP = UNKNOWN.'
            return None
        else:
            gap=diff(E)
            print 'LOWEST GAP = ',gap
            return gap

    def show_flow(self,graph=None,**kwargs):
        '''
        Show flow and rescale ylim.
        '''
        super(RGobj_ABS,self).show_flow(graph=graph,**kwargs)
        if graph!=None:
            vmin,vmax=0,self.flow[-1].max()
            span=vmax-vmin+0.01
            graph[1].set_ylim(vmin,vmax+0.1*span)

class RGobj_Spec(RGobj):
    '''
    Energy Spectrum.
    '''
    def __init__(self,label='Spectrum',nlevel=10000,displaymode='spike',rescale=True,*args,**kwargs):
        super(RGobj_Spec,self).__init__(label,*args,**kwargs)
        self.nlevel=nlevel
        self.setting['displaymode']=displaymode
        self.setting['rescale']=rescale

    def update(self,mval):
        '''
        Update the flow.
        '''
        #filer out None value and update flow.
        if None in mval:
            return
        mval=concatenate(mval)
        self.flow.append(mval)
        print mval[:3]

    def get_expect(self,HN,*args,**kwargs):
        '''
        Get spectrum.

        HN:
            an instance of RGHamiltonian.
        '''
        E=sort(HN.E)[:self.nlevel]
        if self.setting['rescale']:
            E=E*HN.rescalefactor
        EG=E.min()
        res=E-EG
        return res

    def __spec_flow__(self,ax,newflow,**kwargs):
        '''
        present flow data in the spectrum form.

        ax:
            the axis to hold line collections.
        elist:
            the new set of data.
        '''
        #set data
        plot_spectrum(el=newflow,x=arange(2.)/2,offset=[self.nval,0.],ax=ax,**kwargs)

    def __spike_flow__(self,ax,elist,wmax=1.1,b=1e-3,**kwargs):
        '''
        show spectrum of the elist.

        ax:
            the axis.
        wmax:
            the maximum energy.
        elist:
            the energy list.
        b:
            broadening, default is 1e-4
        '''
        wlist=linspace(-wmax,wmax,5000)
        delta_peak_show=gaussian(x=wlist[...,newaxis],mean=elist[newaxis,...],b=b).sum(axis=-1)
        ax.plot(wlist,delta_peak_show,**kwargs)

    def show_flow(self,graph=None,scale=None,**kwargs):
        '''
        Show flow.

        graph:
            the graph tuple (fig,ax,pls). default is None.
        scale:
            the scale(RGScale instance) on which flow is plotted.
        '''
        mode=self.setting['plotmode']
        displaymode=self.setting['displaymode']
        if (mode=='even' and self.nval%2==1) or (mode=='odd' and self.nval%2==0):
            return
        Ntick=5

        #prepair figures
        if graph:
            fig,ax,pls=graph
        else:
            ax=gca()

        #set x ticks.
        if displaymode=='spike':
            ax.set_xlabel('Energy')
            ax.set_xlim(0,1.1)
            self.__spike_flow__(ax=ax,elist=self.cval)
        else:
            flow=array(self.flow)
            scale=arange(len(flow))
            ax.set_xlabel('Step')
            ax.set_xticks(scale)
            ax.set_xticklabels(scale)
            #set data
            if displaymode=='spec':
                self.__spec_flow__(ax,self.cval)
            elif displaymode=='plot':
                ax.plot(scale,flow,lw=2,color='k')
        if graph:
            fig.canvas.draw()

class RGobj_Energy(RGobj):
    '''
    Physics quantity of Energy:
        <H> = Tr(rho*H)/Z
    '''
    def __init__(self,label='Energy',*args,**kwargs):
        super(RGobj_Energy,self).__init__(label,requirements=None,*args,**kwargs)

    def get_expect(self,HN,beta,*args,**kwargs):
        '''
        Get expectation value of sussceptibility chi.

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        E=HN.E_rescaled
        EG=E.min()
        if ndim(beta)==0:
            rho=exp(-beta*(E-EG))
            Z=sum(rho)
            em=sum(E*rho)/Z
        else:
            beta=array(beta)
            rho=exp(-beta[:,newaxis]*(E-EG)[newaxis,:])
            Z=sum(rho,axis=1)
            em=sum(E[newaxis,...]*rho,axis=1)/Z
        return em

class RGobj_Entropy(RGobj):
    '''
    Physics quantity Entropy:
        S/k_B = beta*<H> + ln(Z)
    '''
    def __init__(self,label='Entropy',requirements=None,*args,**kwargs):
        super(RGobj_Entropy,self).__init__(label,requirements=requirements,*args,**kwargs)
        self.subtract_env=True

    def get_expect(self,HN,beta,*args,**kwargs):
        '''
        Get expectation value of sussceptibility chi.

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        E=HN.E_rescaled
        EG=E.min()
        E=E-EG
        if ndim(beta)==0:
            rho=exp(-beta*E)
            Z=sum(rho)
            em=sum(E*rho)/Z
        else:
            beta=array(beta)
            rho=exp(-beta[:,newaxis]*E[newaxis,:])
            Z=sum(rho,axis=1)
            em=sum(E[newaxis,...]*rho,axis=1)/Z
        S=beta*em+log(Z)
        return S

class RGobj_Aw(RGobj):
    '''
    Spectrum function.
        A=sum(|CF_rr'|^2(exp(-beta*Er)+exp(-beta*Er')))/Z
    '''
    def __init__(self,wlist,label='Aw',eta=1e-2,*args,**kwargs):
        super(RGobj_Aw,self).__init__(label,*args,**kwargs)
        self.wlist=wlist
        self.eta=eta
    
    @property
    def wmin(self):
        '''
        the minimum energy scale.
        '''
        return self.wlist.min()

    @property
    def wmax(self):
        '''
        the maximum energy scale.
        '''
        return self.wlist.max()

    def get_expect(self,HN,beta,*args,**kwargs):
        '''
        Get expectation value of A(w).

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        E=HN.E_rescaled
        EG=E.min()
        E=E-EG
        Nc=len(HN.CF)

        rho=exp(-beta*E)
        RM=rho[...,newaxis]+rho[newaxis,...] #matrix of exp(-beta*Ei)+exp(-beta*Ej)
        DEM=E[newaxis,...]-E[...,newaxis]   #matrix of Ej-Ei
        MMRM=[(HN.CF[i]*HN.CF[i].conj()).astype(float64)*RM for i in xrange(Nc)]
        Z=sum(rho)
        Am=[[-1/pi/Z*sum(MMRM[i]*(1./(w-DEM+1j*self.eta)).imag) for w in self.wlist] for i in xrange(Nc)]
        return Am

class RGobj_Cv(RGobj):
    '''
    Physics quantity Cv(equal volume specific heat):
        Cv/k_B = beta^2*(<H^2> - <H>^2)
        due to the hard-convergence of H^2 term, numerical differenciation of the entropy is more appropriate(Costi 1994).
    '''
    def __init__(self,label='Cv',requirements=None,*args,**kwargs):
        super(RGobj_Cv,self).__init__(label,requirements=requirements,*args,**kwargs)
        self.subtract_env=True

    def get_expect(self,HN,beta,*args,**kwargs):
        '''
        Get expectation value of sussceptibility chi.

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        E=HN.E_rescaled
        EG=E.min()
        E=E-EG
        if ndim(beta)==0:
            rho=exp(-beta*E)
            Z=sum(rho)
            em=sum(E*rho)/Z
            e2m=sum(E**2*rho)/Z
        else:
            beta=array(beta)
            rho=exp(-beta[:,newaxis]*E[newaxis,:])
            Z=sum(rho,axis=1)
            em=sum(E[newaxis,...]*rho,axis=1)/Z
            e2m=sum((E**2)[newaxis,...]*rho,axis=1)/Z
        Cv=beta**2*(e2m-em**2)
        return Cv

