#!/usr/bin/python
from numpy import *

from tba.hgen import SuperSpaceConfig,sz,sx,sy

__all__=['NRGImp','AndersonImp','KondoImp','SCImp','NullImp']

class NRGImp(object):
    '''
    An standard application for NRG to run.
    '''
    def __init__(self,spaceconfig,H0):
        self.spaceconfig=spaceconfig
        self.H0=H0

    def show_spectrum(self,**kwargs):
        '''
        Show spectrum function Delta(w).
        '''
        wlist=linspace(-1.5,1.5,200)
        dl=array([self.Delta(w) for w in wlist])
        plot(wlist,dl,**kwargs)

class KondoImp(NRGImp):
    '''
    Kondo Impurity for NRG.
    '''
    def __init__(self,J=0.05,ed=0.,Bz=0.):
        self.J=J
        self.Bz=Bz
        self.ed=ed
        scfg=SuperSpaceConfig([1,2,1,1])
        H0=ed*identity(2)+Bz*sz
        super(KondoImp,self).__init__(scfg,H0)

class AndersonImp(NRGImp):
    '''
    Anderson Impurity for NRG.

    ed/Bz:
        on-side terms.
    U:
        interaction strength.
    '''
    def __init__(self,ed=0.,U=0.,Bz=0.):
        scfg=SuperSpaceConfig([1,2,1,1])
        self.U=U
        self.H0=ed*identity(2)+Bz*sz
        super(AndersonImp,self).__init__(scfg,H0)

    @property
    def ed(self):
        '''on-site energy.'''
        return trace(self.H0)/2.

    @property
    def Bz(self):
        '''magnetic term.'''
        return trace(self.H0.dot(sz))/2.

    def __str__(self):
        return '''Anderson Impurity
    ed: %s
    U: %s
    Bz: %s
    H0: %s
    '''%(self.ed,self.U,self.Bz,self.H0)

class SCImp(NRGImp):
    '''
    Superconducting impurity for NRG.
    '''
    def __init__(self,ed=0.,U=1.,Bz=0.):
        self.U=U
        self.ed=ed
        self.Bz=Bz
        scfg=SuperSpaceConfig([1,2,1,1])

        print 'check!'
        self.H0=(ed+U/2.)*sz+(U/2.+Bz)*identity(2)
        super(SCImp,self).__init__(scfg,self.H0)

class NullImp(NRGImp):
    '''
    Null Impurity for NRG.
    '''
    def __init__(self):
        scfg=SuperSpaceConfig([1,2,1,1])
        H0=zeros([2,2])
        super(NullImp,self).__init__(scfg,H0)

