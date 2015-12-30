from numpy import *
from scipy.sparse import csr_matrix
from hexpand.hexpand import op2expand
from hgen.spaceconfig import SuperSpaceConfig
from hgen.oplib import op_c,op_cdag,op_simple_onsite,op_M,op_D
from core.phcore import OpArray
from core.utils import spin_dict

def rgop_n(i,spinconfig='up'):
    '''
    number of i-th site.

    i:
        site index.
    spinconfig:
        the spin configuration, refer spindict for more detail.
    '''
    spaceconfig=SuperSpaceConfig([1,2,1,1])
    n=0
    for s1,s2,sfactor in spin_dict[spinconfig]:
        opcd=op_cdag(spaceconfig=spaceconfig,index=s1)().view(OpArray)
        opc=op_c(spaceconfig=spaceconfig,index=s2)().view(OpArray)
        n=n+dot(opcd,opc)
    n.label='n%s%s'%(i,spinconfig)
    return op2expand(n,expansion_method='expand',starting_level=i+1)

def rgop_d(i,spinconfig='up'):
    '''
    superconducting order of i-th site.

    i:
        site index.
    spinconfig:
        the spin configuration, refer spindict for more detail.
    '''
    spaceconfig=SuperSpaceConfig([1,2,1,1])
    n=0
    for s1,s2,sfactor in spinconfig:
        opcd1=op_cdag(spaceconfig=spaceconfig,index=s1)().view(OpArray)
        opcd2=op_cdag(spaceconfig=spaceconfig,index=s2)().view(OpArray)
        n=n+dot(opcd1,opcd2)
    n.label='d%s%s'%(i,spinconfig)
    return op2expand(n,expansion_method='expand',starting_level=i+1)

def rgop_cd(i,spinindex=0):
    '''
    get the c^dag operator for specific site.

    i:
        the site index.
    spinindex:
        the spin index.
    '''
    spaceconfig=SuperSpaceConfig([1,2,1,1])
    fd=op_cdag(spaceconfig=spaceconfig,index=spinindex)().view(OpArray)
    fd.label='f%s%sd'%(i,'up' if spinindex==0 else 'dn')
    return op2expand(fd,expansion_method='expand',starting_level=i+1)

def rgop_c(i,spinindex=0):
    '''
    get the c^dag operator for specific site.

    i:
        the site index.
    spinindex:
        the spin index.
    '''
    spaceconfig=SuperSpaceConfig([1,2,1,1])
    fd=op_c(spaceconfig=spaceconfig,index=spinindex)().view(OpArray)
    fd.label='f%s%s'%(i,'up' if spinindex==0 else 'dn')
    return op2expand(fd,expansion_method='expand',starting_level=i+1)

def rgop_c3(i,spinindex):
    r'''
    get NRG operator $c_{i\sigma}c_{i\bar{\sigma}}^\dag c_{i\bar{\sigma}}$
    '''
    spaceconfig=SuperSpaceConfig([1,2,1,1])
    f1=op_c(spaceconfig=spaceconfig,index=spinindex)()
    f2=op_cdag(spaceconfig=spaceconfig,index=1-spinindex)()
    fff=op_c(spaceconfig=spaceconfig,index=1-spinindex)()
    FFF=f1.dot(f2.dot(fff)).view(OpArray)
    FFF.label='fff%s%s'%(i,'up' if spinindex==0 else 'dn')
    return op2expand(FFF,expansion_method='expand',starting_level=i+1)

#predefined operators
spaceconfig=SuperSpaceConfig([1,2,1,1])
I1=identity(spaceconfig.hndim).view(OpArray)
cupd=op_cdag(spaceconfig=spaceconfig,index=0)().view(OpArray)
cup=op_c(spaceconfig=spaceconfig,index=0)().view(OpArray)
cdnd=op_cdag(spaceconfig=spaceconfig,index=1)().view(OpArray)
cdn=op_c(spaceconfig=spaceconfig,index=1)().view(OpArray)
nup=op_simple_onsite(label='nup',spaceconfig=spaceconfig,index=0)().view(OpArray)
ndn=op_simple_onsite(label='ndn',spaceconfig=spaceconfig,index=1)().view(OpArray)
g=dot(cupd,cdnd)
sx=op_M(spaceconfig,direction='x')().view(OpArray)
sy=op_M(spaceconfig,direction='y')().view(OpArray)
sz=op_M(spaceconfig,direction='z')().view(OpArray)
op_sgn=(I1-2*nup)*(I1-2*ndn)

I1_s=csr_matrix(I1)
cupd_s=csr_matrix(cupd)
cup_s=csr_matrix(cup)
cdnd_s=csr_matrix(cdnd)
cdn_s=csr_matrix(cdn)
nup_s=csr_matrix(nup)
ndn_s=csr_matrix(ndn)
g_s=csr_matrix(g)
sx_s=csr_matrix(sx)
sy_s=csr_matrix(sy)
sz_s=csr_matrix(sz)
op_sgn_s=csr_matrix(op_sgn)
