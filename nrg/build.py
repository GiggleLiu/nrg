import numpy as np
import scipy.sparse as sps
import re
from pymps.construct.opstring import OpUnit
from pymps.construct.opstringlib import opunit_cdag, opunit_c, opunit_N, opunit_Z, insert_Zs
from pymps.ansatz.mpo import OPC2MPO
from pymps.blockmarker import SimpleBMG

__all__ = ['BuildingSpace', 'build_mpo', 'mat2opc']

class BuildingSpace(object):
    '''
    generate a set of unit operators like cdag and c,
    to facilitate building hamiltonians.

    Args:
        spaceconfig (SpaceConfig): Hilbert sapce configuration for a single site.
    '''
    def __init__(self, spaceconfig=[1,2,1]):
        self.spaceconfig = spaceconfig

    def __getattr__(self, name):
        op_str_list = [r'cdag(\d)', r'c(\d)', r'n(\d)', r'z']
        mat_str_list = [r'CDAG(\d)', r'C(\d)', r'N(\d)', r'Z']
        constructor_list = [opunit_cdag, opunit_c, opunit_N, opunit_Z]
        for s, S, constr in zip(op_str_list, mat_str_list, constructor_list):
            mop = re.match(s,name)
            mmat = re.match(S,name)
            if mop:
                res = constr(self.spaceconfig, *mop.groups())
                return res
            if mmat:
                res = constr(self.spaceconfig, *mmat.groups())
                res = sps.csr_matrix(res.data)
                res.eliminate_zeros()
                return res
        raise AttributeError('Operator %s not found'%name)


def build_mpo(spaceconfig, h_impurity, chain, quantum_numbers='QM'):
    '''
    build a Wilson chain MPO from a Chain instance.
    
    Args:
        spaceconfig (SpaceConfig): Hilbert sapce configuration for a single site.
        h_impurity (2darray): hamiltonian on the impurity site, its dimension can be 0-1 sites.
        chain (:obj:`Chain`): Wilson chain generated in nrgmap.
        quantum_numbers ('Q'|'M'|'P'|'R'|combinations of them): good quantum numbers.
            * Q: charge
            * M: spin
            * P: charge parity
            * R: spin parity

    Returns:
        :obj:`MPO`: matrix product operator instance.
    '''
    hndim = spaceconfig.hndim
    ndim = spaceconfig.ndim
    bspace = BuildingSpace(spaceconfig)
    basis = [getattr(bspace,'c%d'%i) for i in range(ndim)]
    basis_diag = [getattr(bspace,'cdag%d'%i) for i in range(ndim)]
    bmg = SimpleBMG(quantum_numbers, spaceconfig=spaceconfig)

    hamiltonian, imp_nsite = mat2opc(h_impurity, bmg=bmg)
    for isite, (ei, ti) in enumerate(zip(chain.elist, chain.tlist)):
        siteindex = isite+imp_nsite
        for i in range(ndim):
            for j in range(ndim):
                if abs(ei[i,j])>1e-15:
                    hamiltonian = hamiltonian+ei[i,j]*basis_diag[i].as_site(siteindex)*basis[j].as_site(siteindex)
                if abs(ti[i,j])>1e-15 and siteindex>0:
                    hamiltonian = hamiltonian+ti[i,j]*basis_diag[i].as_site(siteindex-1)*basis[j].as_site(siteindex)
                    hamiltonian = hamiltonian-ti[i,j]*basis[i].as_site(siteindex-1)*basis_diag[j].as_site(siteindex)
    insert_Zs(hamiltonian, spaceconfig)
    mpo = OPC2MPO(hamiltonian, method='direct', bmg=bmg)
    return mpo

def mat2opc(h_impurity, bmg):
    '''
    generate operator unit/string/collection representation from matrix representation.
    In NRG application, it is used to decode impurity hamiltonian matrix.

    Args:
        h_impurity (2darray): hamiltonian on the impurity site, its dimension can be 0-1 sites.
        bmg (:obj:`BlockMarkerGenerator`): block marker generation handler.

    Returns:
        tuple(OpUnit|OpString|OpCollection|0, int): impurity hamiltonian and number of sites in impurity.
    '''
    hndim = bmg.spaceconfig.hndim
    if h_impurity.shape[1] == 1:
        return 0, 0
    elif h_impurity.shape[1] == hndim:
        return OpUnit('h0', h_impurity, siteindex=0), 1
    elif h_impurity.shape[1] == hndim**2:
        raise
    else:
        raise


