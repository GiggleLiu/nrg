from numpy import *

from tba.hgen import SuperSpaceConfig,sz,sx,sy
from tba.hgen import op_U,Operator

__all__=['Impurity','AndersonImp','KondoImp','SC2Anderson','NullImp','FreeImp']

class Impurity(object):
    '''
    Impurity.
    '''
    def __init__(self,spaceconfig,H0):
        self.spaceconfig=spaceconfig
        self.H0=H0

    def get_interaction(self):
        '''
        Get the interaction term.
        
        Return:
            <Operator>, the interaction part.
        '''
        raise NotImplementedError()

class KondoImp(Impurity):
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

class AndersonImp(Impurity):
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
        H0=ed*identity(2)+Bz*sz
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

    def get_interaction(self):
        '''
        Get the interaction operator.
        '''
        return self.U*op_U(self.spaceconfig)

class NullImp(Impurity):
    '''
    Null Impurity for NRG.
    '''
    def __init__(self):
        scfg=SuperSpaceConfig([1,2,1,1])
        super(NullImp,self).__init__(scfg,H0=None)

    def get_interaction(self):
        '''
        Get the interaction operator.
        '''
        return Operator('',spaceconfig=self.spaceconfig)

class FreeImp(Impurity):
    '''
    Free Impurity for NRG, 0 - energy single site.
    '''
    def __init__(self,mu=0.):
        scfg=SuperSpaceConfig([1,2,1,1])
        H0=-mu*identity(scfg.ndim)
        super(NullImp,self).__init__(scfg,H0)

    def get_interaction(self):
        '''
        Get the interaction operator.
        '''
        return Operator('Null',self.spaceconfig)

def SC2Anderson(ed=0.,U=1.,Bz=0.):
    '''
    Transforn a superconducting Anderson problem to normal base.
    '''
    scfg=SuperSpaceConfig([1,2,1,1])
    return AndersonImp(ed=U/2.+Bz,U=-U,Bz=(ed+U/2.))
