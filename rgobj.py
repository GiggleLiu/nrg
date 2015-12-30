#!/usr/bin/python
from scipy import *
from matplotlib.pyplot import *
from scipy.interpolate import splrep,splev
from hgen.spaceconfig import SuperSpaceConfig
from hgen.oplib import op_M
from core.matrixlib import bcast_dot
from core.utils import plot_spectrum,sx,sy,sz
from core.phcore import Mobj,OpArray
from core.mathlib import log_gaussian,gaussian,lorenzian
from hexpand.hexpand import op2expand
from rgoplib import rgop_c,rgop_n
from scipy.sparse import issparse
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
import pdb,time,warnings

warnings.simplefilter('once',UserWarning)

class RGRequirement(object):
    '''
    specify a requirement.

    label:
        the label.
    tp:
        the type, `op`/`attr`/`bool`
    islist:
        get the list if True(False by default).
    beforetrunc:
        get it before truncation if True(False by default).
    info:
        more information.
    '''
    def __init__(self,label,tp,islist=False,beforetrunc=False,info=None):
        self.label=label
        self.tp=tp
        self.islist=islist
        self.beforetrunc=beforetrunc
        if info is None:
            self.info={}
        else:
            self.info=info

    def __str__(self):
        return self.label+'('+self.tp+')'

class RGobj(Mobj):
    '''
    Measuring objects used for NRG.

    label:
        the label.
    requirements:
        the required variables.
    plotmode: plot mode,
    NOTE:
        4 plot modes are usable, you can specify it by accessing self.setting['plotmode'].
        * mixing -> mixing data with neghbors to avoid odd-even steps.
        * even -> get even part only.
        * odd -> get odd part only.
        * both -> get original data.
    '''
    def __init__(self,label,requirements=None,plotmode='mixing',*args,**kwargs):
        super(RGobj,self).__init__(label,*args,**kwargs)
        self.requirements=[]
        self.subtract_env=False
        if not requirements is None:
            self.requirements=requirements
        self.setting['plotmode']=plotmode

    def show_flow(self,graph,xdata,**kwargs):
        '''
        Get suitible scale,data to plot.

        scale:
            the x-axis
        '''
        if graph:
            fig,ax,pls=graph
        else:
            ax=gca()
            pls=ax.plot([],[],**kwargs)
        ax.set_xscale('log')
        rcParams['xtick.labelsize']=14
        rcParams['ytick.labelsize']=14
        rcParams['axes.labelsize']=14

        mode=self.setting['plotmode']
        flow=array(self.flow)
        if xdata is None:
            if mode=='mixing':
                warnings.warn('xdata should be specified to use mixing mode for display flow data. switching to plot mode `both`.')
                mode='both'
            xdata=arange(self.nval)
        elif mode=='even':
            xdata,flow=xdata[::2],flow[::2]
        elif mode=='odd':
            xdata,flow=xdata[1::2],flow[1::2]
        elif mode=='mixing':
            if len(flow)>2:
                #smearing flow-data
                flowm1=roll(flow,1,axis=0)
                flowm1[0]=flow[0]
                flowp1=roll(flow,-1,axis=0)
                flowp1[-1]=flow[-1]

                tlm1=roll(xdata,1)
                tlm1[0]=xdata[0]
                tlp1=roll(xdata,-1)
                tlp1[-1]=xdata[-1]
                flow2=copy(flow)
                flow=0.5*(flow+flowm1+(flowp1-flowm1)*((xdata-tlm1)/(tlp1-tlm1)))
        else:
            warnings.warn('plot mode unknown @%s'%self)

        pls[0].set_xdata(xdata)  #update data
        pls[0].set_ydata(flow)  #update data
        ax.set_ylabel(self.label)
        ax.set_xlabel(r'$k_BT$')
        ax.set_ylim(0,0.25)
        ax.set_xlim(xdata.min(),xdata.max())
        if graph:
            fig.canvas.draw()

    def update(self,mval):
        '''
        Update the flow.
        '''
        #filer out None value and update flow.
        if mval[0] is None:
            return
        mval=mean(mval,axis=0)
        self.flow.append(mval)

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

    label:
        the label, by default it is 'chi'.
    mode:
        's' -> spin is a good quantum number for hamiltonian.
        'on' -> use operator mode with operators defined on all sites.
        'o1' -> use operator mode with operators defined on first site.
    spaceconfig:
        it should be provided if it runs in 'o' mode.
    '''
    def __init__(self,label='chi',mode='s',spaceconfig=None,*args,**kwargs):
        super(RGobj_Tchi,self).__init__(label,*args,**kwargs)
        self.mode=mode
        if mode=='s':
            self.subtract_env=True
        elif mode=='o1':
            self.__init_ops__(spaceconfig,'expand')
        elif mode=='on':
            self.subtract_env=True
            self.__init_ops__(spaceconfig,'duplicate')

    def __init_ops__(self,spaceconfig,mode):
        '''
        Initialize required oprators,

        spaceconfig:
            the configuration of space.
        '''
        sz=op_M(spaceconfig,direction='sz')()/2
        sz=sz.view(OpArray)
        sz.label='sz'
        sz=op2expand(sz,expansion_method=mode,starting_level=1)
        self.requirements.append(RGRequirement(sz.label,tp='op',info={'kernel':sz}))

    def get_expect(self,HN,*args,**kwargs):
        '''
        Get the expectation value of Tchi.

        HN:
            the hamiltonian instance.
        '''
        if self.mode=='o1' or self.mode=='on':
            return self.__get_expect_o__(HN,*args,**kwargs)
        else:
            return self.__get_expect_s__(HN,*args,**kwargs)

    def __get_expect_s__(self,HN,beta,*args,**kwargs):
        '''
        Get expectation value of sussceptibility chi.
        In the Q,S,m revserved case, we can use the following relation ->
            chi = S(S+1)/3 x (2*S+1) - sum(sz**2 for sz in range(-S,S))

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        E=HN.E_rescaled
        EG=E.min()
        Z=sum(exp(-beta*(E-EG)))
        mval=0.
        szm=0.  #mean of Sz
        if HN.reserved_quantities=='QSm':
            rescalefactor=HN.rescalefactor
            for block in HN.blocks:
                rho=sum(exp(-beta*(block.E*rescalefactor-EG)))
                ind=block.indexer
                #ind is (Q,Sz), each energy represents (2*S+1) dengeneracy.
                S=ind[1]
                mval+=rho*(S*(S+1)*(2*S+1)/3)/Z
        elif HN.reserved_quantities=='m' or HN.reserved_quantities=='Nm':
            bmarker=HN.block_marker
            s=0
            for i in xrange(bmarker.nblock2):
                Ei=bmarker.extract_block(E,i,j=None,trunced=True)
                rho=sum(exp(-beta*(Ei-EG)))
                #ind is Sz
                Q,sz=HN.__decode_token__(HN.ind2token(i))
                mval+=rho*sz**2/Z
                szm+=rho*sz/Z
        else:
            raise Exception('Error','matrix with reserved quantities %s not supported in RGObj_Tchi!'%HN.reserved_quantities)
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
        opsz=HN.collect_op('sz')
        opsz2=opsz.dot(opsz)
        szm=HN.get_expect('sz',T=1./beta,*args,**kwargs).real
        sz2m=HN.get_expect(opsz2,T=1./beta,*args,**kwargs).real
        return (sz2m-szm**2)

class RGobj_A(RGobj):
    '''
    Spectrum function for specific site.
    a flow element is defined as [w,A(w),broadening]

    label:
        the label.
    spinindex/site:
        the specific site and spinindex to measure.
    T:
        the temperature, default is 0
    wfactor:
        the relative w to measure with respect to the energy scale at each expansion.
    bfactor:
        broadening factor.
    '''
    def __init__(self,label='A',spinindex=0,site=0,T=0,wfactor=2.,logscale=False,continuate=None,*args,**kwargs):
        super(RGobj_A,self).__init__(label,*args,**kwargs)
        self.site=site
        self.mode=None
        self.spinindex=spinindex
        self.setting.update({
            'wfactor':wfactor,
            'bfactor':0.5,
            'smearing_method':'gaussian',
            'continuate':continuate,
        })
        self.T=T
        self.stopped=False   #to tell if it will continue updating spectrums(used in finite temperature).
        self.logscale=logscale

    def setup_normal(self):
        '''
        set mode to `normal` - choose a w to update at each iteration.
        '''
        self.mode='normal'
        rop=rgop_c(self.site,self.spinindex)
        self.requirements.append(RGRequirement(rop.label,tp='op',info={'kernel':rop}))

    def setup_sigma(self):
        '''set mode to `sigma` - use bulla's sigma method to get A.'''
        self.mode='sigma'
        rop=rgop_c(self.site,self.spinindex)
        self.requirements.append(RGRequirement(rop.label,tp='op',info={'kernel':rop}))


    def setup_rho(self,N):
        '''
        set mode to `rho` - update at the last iteration, done by extracting information from reduced density matrix.

        N:
            the point to decided the lowest energy.
        '''
        self.mode='rho'
        rop=rgop_c(self.site,self.spinindex)
        self.requirements.append(RGRequirement('E_rescaled',tp='attr',islist=True))
        self.requirements.append(RGRequirement('U',tp='attr',islist=True))
        self.requirements.append(RGRequirement('blocks',tp='attr',islist=True))
        self.requirements.append(RGRequirement(rop.label,tp='op',info={'kernel':rop},islist=True))
        self.setting['rho-measurepoint']=N
        self.cache['flist']=[]
        self.cache['elist']=[]
        self.cache['rslist']=[]

    def set_smearing(self,smearing_method,b=None):
        '''
        Set Smearing method.

        smearing_method:
            method for smearing peaks.
            'gaussian'(default): use gaussian peak.
            'log_gaussian': use log_gaussian peak(symmetric function).
            'lorenzian': use lorenzian peak.
        b:
            the broadening factor.
        '''
        self.setting['smearing_method']=smearing_method
        if b!=None:
            self.setting['bfactor']=b

    def __get_w__(self,wunit):
        '''
        Get w-list and b for evaluation
        '''
        w=self.setting['wfactor']*wunit
        smearing_method=self.setting['smearing_method']
        #gate value to view energy the same(to detect degeneracy).
        gate=wunit*1e-4
        ws=array([-w,w])
        return ws

    def __smear__(self,x,mean,b=None):
        '''smear datas.
        
        x:
            a list of w.
        mean:
            a list of energy.
        b:
            broadening
        '''
        smearing_method=self.setting['smearing_method']
        bfactor=b if b!=None else self.setting['bfactor']
        b=array([bfactor]*len(x)).reshape(x.shape)
        if smearing_method=='log_gaussian':
            res=log_gaussian(x,mean,b.reshape(x.shape))
            return res
        elif smearing_method=='gaussian':
            return gaussian(x,mean,b)
        elif smearing_method=='lorenzian':
            return lorenzian(x,mean,b)
        else:
            warnings.warn('Unknow smearing method -set to default `gaussian`')
            return gaussian(x,mean,b)

    def __get_expect_0K__(self,spinindex,HN,beta,show_spec=True,*args,**kwargs):
        '''
        Get spectrum for zero temperature(with block spectrum only).

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        show_spec:
            show spectrum within each recursion.
        '''
        rescalefactor=HN.rescalefactor
        label='f%s%s'%(self.site,'up' if spinindex==0 else 'dn')
        E=HN.E_rescaled
        EG=E.min()
        gate=rescalefactor*1e-4
        Z=sum((E-EG)<gate)
        A=0.
        ws=self.__get_w__(rescalefactor)
        fop=HN.cov_ops[label].data
        HN.scatter_op(label,fop,isdiag=False)
        for block in HN.blocks:
            ind=block.indexer
            if block.isnull:
                continue
            Eb=block.E[0]*rescalefactor
            NE=sum(abs(Eb-EG)<gate)
            if NE==0:
                continue
            #trace back source block to get M[0,r]
            #note that the current block is connected by actting cd to source block.
            q,m=HN.__decode_token__(ind)
            block_from=HN.tgetblock(HN.__encode_token__(q-1,m-(0.5-spinindex)))
            E_from=block_from.E*rescalefactor-EG
            f_from=block_from.cov_ops[label].get()
            pdb.set_trace()

            block_to=HN.tgetblock(HN.__encode_token__(q+1,m+(0.5-spinindex)))
            E_to=block_to.E*rescalefactor-EG
            f_to=block.cov_ops[label]

            for ie in xrange(NE):
                #get c-matrices
                Mr02=(f_from[:,ie].conj()*f_from[:,ie]).real
                M0r2=(f_to[ie,:].conj()*f_to[ie,:]).real

                A+=(Mr02*self.__smear__(x=ws[...,newaxis],mean=-E_from)).sum(axis=-1)/Z
                A+=(M0r2*self.__smear__(x=ws[...,newaxis],mean=E_to)).sum(axis=-1)/Z

        #show spectrum for debug
        if False:
            if self.cache.has_key('fig'):
                fig=self.cache.get('fig')
            else:
                fig=figure()
                self.cache['fig']=fig
            fig.clf()
            self.show_spec(E_to,b=0.39*wunit,wmax=10*wunit)
            self.show_spec(-E_from,b=0.39*wunit,wmax=10*wunit)
            self.show_spec(E-EG,b=0.39*wunit,wmax=10*wunit)
            axvline(w)
            axvline(-w)
            show()
            legend(['E_to','E_from','E_all'])
            pdb.set_trace()
        return array([ws,A])

    def __get_expect_0K2__(self,spinindex,HN,beta,*args,**kwargs):
        '''
        Get spectrum for zero temperature with whole Energy spectrum.

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        rescalefactor=HN.rescalefactor
        gate=rescalefactor*1e-4
        #specific energy spectrum to extract
        ws=self.__get_w__(rescalefactor)
        #get the operator for c^dag
        label='f%s%s'%(self.site,'up' if spinindex==0 else 'dn')
        #gate value to view energy the same(to detect energy degeneracy).

        E=HN.E_rescaled
        opmatrix=HN.collect_op(label)
        if issparse(opmatrix):
            opmatrix=opmatrix.toarray()
        EG=E.min()

        EIND=where((E-EG)<gate)
        Z=len(EIND[0])
        A=0.
        for i in EIND[0]:
            Mr02=opmatrix[:,i].conj()*(opmatrix[:,i])
            M0r2=opmatrix[i,:].conj()*(opmatrix[i,:])
            A+=(Mr02*self.__smear__(x=ws[...,newaxis],mean=-(E-EG))).sum(axis=-1)/Z
            A+=(M0r2*self.__smear__(x=ws[...,newaxis],mean=E-EG)).sum(axis=-1)/Z
        return array([ws,A.real])

    def __get_expect_finite__(self,spinindex,HN,beta,*args,**kwargs):
        '''
        Get spectrum for zero temperature with whole Energy spectrum.

        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        if self.stopped:
            return None

        rescalefactor=HN.rescalefactor
        ws,b=self.__get_ws__(rescalefactor)
        w=self.setting['wfactor']*wunit
        label='f%s%s'%(self.site,'up' if spinindex==0 else 'dn')

        E=HN.E*rescalefactor
        opmatrix=HN.collect_op(label)
        EG=E.min()
        rho=exp(-(E-EG)/self.T)
        Z=sum(rho)
        #below the temperature
        if w<self.T and False:
            b=self.T/w/4.
            self.stopped=True
        else:
            #decide energy spectrum and smearing factor
            ws=array([-w,w])

            opmatrix=HN.collect_op(label)
            opmatrix=(opmatrix*opmatrix.conj()).real
            distri=rho[...,newaxis]+rho[newaxis,...]
            delta_peak=self.__smear__(x=ws[...,newaxis,newaxis],mean=E[:,newaxis]-E[newaxis,:],b=b)*distri
            A=(opmatrix*delta_peak).sum(axis=(-1,-2))/Z
        return array([ws,A])

    def __get_expect_rho__(self,spinindex,HN,beta,*args,**kwargs):
        '''
        Get spectrum for zero temperature with whole Energy spectrum.

        spinindex:
            0 for spin up and 1 for spin down.
        HN:
            an instance of RGHamiltonian.
        beta:
            the inverse of temperature.
        '''
        rescalefactor=HN.rescalefactor
        gate=rescalefactor*1e-4
        #get the operator for c^dag
        label='f%s%s'%(self.site,'up' if spinindex==0 else 'dn')
        #gate value to view energy the same(to detect energy degeneracy).
        E=HN.E_rescaled
        opmatrix=HN.collect_op(label)

        #cache it for later usage.
        self.cache['flist'].append(opmatrix)
        self.cache['elist'].append(E)
        self.cache['rslist'].append(rescalefactor)

        #measure
        if HN.N==self.setting['rho-measurepoint']:
            flist=self.cache['flist']
            elist=self.cache['elist']
            rslist=self.cache['rslist']
            ndata=len(elist)
            rholist=HN.rho_red_list()[2:]   #pity here that we do not start from zero!
            Alist=[]
            wlist=[]
            for i in xrange(ndata):
                M=flist[i]
                Ei=elist[i]
                rho=rholist[i]
                rescalefactor=rslist[i]
                #specify energy spectrum to extract
                ws=self.__get_w__(rescalefactor)
                delta_peak=self.__smear__(x=ws[...,newaxis,newaxis],mean=Ei[:,newaxis]-Ei[newaxis,:])
                A=trace(bcast_dot(delta_peak*(dot(rho,M)+dot(M,rho)),M.T.conj()),axis1=-1,axis2=-2).real
                wlist=append(wlist,ws)
                Alist=concatenate([Alist,A],axis=0)
            rightorder=argsort(wlist)
            return wlist[rightorder],Alist[rightorder]

    def show_flow(self,graph,xdata):
        '''
        Get suitible scale,data to plot.

        scale:
            the x-axis
        '''
        if graph:
            fig,ax,pls=graph
        else:
            ax=gca()
            pls=ax.plot([],[],**kwargs)
        rcParams['xtick.labelsize']=14
        rcParams['ytick.labelsize']=14
        rcParams['axes.labelsize']=14

        if self.mode=='rho':
            if len(self.flow)>0:
                data=self.flow[-1]
            else:
                data=zeros(2)
        plotmode=self.setting['plotmode']
        if plotmode=='even':
            sl=slice(None,None,2)
        elif plotmode=='odd':
            sl=slice(1,None,2)
        else:
            sl=slice(None,None)
        wdata,A=data[sl,0].ravel(),data[sl,1].ravel()

        sind=argsort(wdata)
        wdata,A=wdata[sind],A[sind,newaxis]
        if self.logscale:
            wscale=log(abs(wdata))/log(10)
            wp=wscale[wdata>0]
            wn=wscale[wdata<0]
            wscale[wdata>0]=wp-wp.min()
            wscale[wdata<0]=wn-wn.max()
            xdata=linspace(wscale.min(),wscale.max(),len(wscale))
        if self.setting['continuate']=='spline' and len(wdata)>2:
            Afunc=spline1d(wdata,A,k=2)
            xdata=linspace(wdata.min(),wdata.max(),10000)
            A=Afunc(xdata)
            #tck=splrep(wdata,A,k=3)
            #xdata=linspace(wdata.min(),wdata.max(),400)
            #A=splev(float64(wdata),tck,der=0)
        else:
            warnings.warn('plot mode unknown @%s'%self)

        pls[0].set_xdata(xdata)  #update data
        pls[0].set_ydata(A)  #update data
        ax.set_ylabel('A')
        ax.set_ylim(0,A.max()+0.01)
        ax.set_xlim(xdata.min(),xdata.max())
        ax.set_xlabel(r'$\omega$',fontsize=20)
        if graph:
            fig.canvas.draw()

    def show_spec(self,elist,ax=None,wmax=1.1,b=1e-3):
        '''
        show spectrum of the elist.

        wmax:
            the maximum energy.
        ax:
            axis
        elist:
            the energy list.
        b:
            broadening, default is 1e-4
        '''
        if ax is None:
            ax=gca()
        wlist=linspace(-wmax,wmax,5000)
        delta_peak_show=self.__smear__(x=wlist[...,newaxis],mean=elist[newaxis,...]).sum(axis=-1)
        ax.plot(wlist,delta_peak_show)

    def get_expect(self,HN,*args,**kwargs):
        '''
        Get the expectation value of Tchi.

        HN:
            the hamiltonian instance.
        '''
        spinindex=self.spinindex
        if self.mode=='rho':
            ms=self.__get_expect_rho__(spinindex,HN,*args,**kwargs)
        elif self.T==0:
            #if isinstance(HN,BlockHamiltonian):
                #ms=self.__get_expect_0K__(spinindex,HN,*args,**kwargs)
            #else:
            ms=self.__get_expect_0K2__(spinindex,HN,*args,**kwargs)
        else:
            ms=self.__get_expect_finite__(spinindex,HN,*args,**kwargs)
        return ms

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
        E=sort(HN.E_rescaled)[:self.datalen+1]
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

