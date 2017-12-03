import numpy as np
from scipy.linalg import eigh, eigvalsh
import pdb
import matplotlib.pyplot as plt
from pymps.construct.opstring import OpUnit
from pymps.construct.opstringlib import opunit_cdag, opunit_c, opunit_N, opunit_Z, insert_Zs
from pymps.ansatz.mpo import OPC2MPO
from pymps.ansatz.plotlib import show_mpo
from pymps.blockmarker import SimpleBMG, BlockMarker
from pymps.tensor import Tensor, BLabel
from pymps.tensor.zero_flux import btdot, is_zero_flux
from pymps.spaceconfig import SuperSpaceConfig
import scipy.sparse as sps
import re

def build_mpo(spaceconfig, h_impurity, chain, quantum_numbers='QM'):
    '''build a wilson chain mpo.'''
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
    hndim = bmg.spaceconfig.hndim
    if h_impurity.shape[1] == 1:
        return 0, 0
    elif h_impurity.shape[1] == hndim:
        return OpUnit('h0', h_impurity, siteindex=0), 1
    elif h_impurity.shape[1] == hndim**2:
        raise
    else:
        raise

def is_compact(t):
    if isinstance(t, list):
        return [is_compact(ti) for ti in t]
    # return all(np.all(l.bm.sort().compact_form().qns == l.bm.qns) for l in t.labels)
    return all(np.all(l.bm.sort().compact_form().nblock == l.bm.nblock) for l in t.labels)

def solve_mpo(mpo, max_keep=600, scaling_factor=1.0):
    '''solve a wilson chain mpo'''
    bm0 = mpo.bmg.bm0
    L = Tensor(np.ones([1,1,1]), labels=[BLabel('u_0',bm0), BLabel('b_0', bm0), BLabel('d_0', bm0)])
    sign_L = [1, -1, -1]
    mpo_sign = [1, 1, -1, -1]
    sign_U = [1,1,-1]
    sign_UD = [-1,1,1]
    sign_H = [1,1,-1,-1,-1]
    sign_UH = [1,-1,-1,-1]
    for l in range(mpo.nsite):
        # build L
        mpocell = mpo.get(l)
        if l!=0:
            mpocell = mpocell.take(opm, axis=0)
        mpocell, (opm,) = mpocell.b_reorder(axes=(3,), return_pm = True)
        opm_r = np.argsort(opm)
        # compact bm is important here
        H = btdot(L, mpocell, sign_L, mpo_sign, mpo.bmg, check_bm=True).chorder([0,2,1,3,4])

        # get eigen values, eigen vectors
        base_index = opm_r[H.shape[-1]-1]
        H_ = H.take(base_index,-1)
        # H_ = H.take(H.shape[-1]-1,-1)
        U, E, UD = H_.svd(2, cbond_str='u_%d'%(l+1), kernel='eigh', signs=[1,1,1,1], bmg=mpo.bmg)
        # print('EG = %s, %s'%(eigvalsh(H_.reshape([H_.shape[0]*H_.shape[1],-1]))[0], E.min()))

        # truncation
        mask = E<=np.sort(E)[min(len(E), max_keep)-1]
        yield {'E':E, 'U':U, 'H':H_, 'mask':mask}
        U, UD = U.take(mask,-1), UD.take(mask,0)
        UD.labels[0] = UD.labels[0].chstr('d_%d'%(l+1))

        # update L to diagonal form, and rescale by a constant
        UH = btdot(U.conj(), H, sign_U, sign_H, mpo.bmg, check_bm=True)
        L = btdot(UH, UD.conj(), sign_UH, sign_UD, mpo.bmg, check_bm=True)
        np.fill_diagonal(L[:,base_index,:], L[:,base_index,:].diagonal()-E.min())
        L = L*scaling_factor
        tensor_list = [U, UD, H, UH, L]
        sign_list = [sign_U, sign_UD, sign_H, sign_UH, sign_L]

def ed_solve(spaceconfig, h_impurity, chain, k=1):
    return _ed_solve(spaceconfig, h_impurity, chain, k=1, step_wise=False)

def ed_itersolve(spaceconfig, h_impurity, chain, k=1):
    return _ed_solve(spaceconfig, h_impurity, chain, k=1, step_wise=True)

def _ed_solve(spaceconfig, h_impurity, chain, k, step_wise):
    # build a chain hamiltonian
    hndim = spaceconfig.hndim
    ndim = spaceconfig.ndim
    bspace = BuildingSpace(spaceconfig)
    basis = [getattr(bspace,'C%d'%i) for i in range(ndim)]
    basis_diag = [getattr(bspace,'CDAG%d'%i) for i in range(ndim)]
    Z = bspace.Z

    def _update(H, ei, ti):
        emat = sps.csr_matrix((hndim, hndim))
        tmat = sps.csr_matrix((hndim**2, hndim**2))
        for i in range(ndim):
            for j in range(ndim):
                if abs(ei[i,j])>1e-15:
                    emat = emat+ei[i,j]*basis_diag[i].dot(basis[j])
                if abs(ti[i,j])>1e-15:
                    tmat = tmat+sps.kron(ti[i,j]*basis_diag[i].dot(Z),basis[j])
        tmat = tmat+tmat.T.conj()
        H = sps.kron(H, sps.eye(hndim)) + sps.kron(sps.eye(H.shape[0]), emat) +sps.kron(sps.eye(H.shape[0]//hndim), tmat)
        return H

    H = h_impurity
    if step_wise:
        EG, VG = sps.linalg.eigsh(H, k=k, which='SA')
        yield {'E':EG,'H':H, 'U':VG}
    for ei, ti in zip(chain.elist, chain.tlist):
        H = _update(H, ei, ti)
        if step_wise:
            EG, VG = sps.linalg.eigsh(H, k=k, which='SA')
            yield {'E':EG,'H':H, 'U':VG}

    if not step_wise:
        EG, VG = sps.linalg.eigsh(H, k=1, which='SA')
        return EG[0], VG[:,0], H

class BuildingSpace(object):
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

def _render_string(ops, sites):
    hndim = ops[0].shape[0]
    h = sps.eye(1)
    last_site = -1
    for op, site in zip(ops,sites):
        h = sps.kron(sps.kron(h, sps.eye(hndim**(site-last_site-1))), op)
        last_site = sites
    return h


def test_ed_solve():
    from nrgmap.chain import Chain
    spaceconfig = SuperSpaceConfig([1,2,1])
    # Anderson Impurity
    h_impurity = np.zeros([4,4])
    h_impurity[3,3] = 1.0
    h_impurity[1,1] = -1.0
    h_impurity[2,2] = -1.0
    h_impurity[0,0] = 1.0

    chain = Chain(tlist=[0.5*np.eye(2), 0.2*np.eye(2)] ,elist=[-0.2*np.eye(2), -0.5*np.eye(2)])
    res = ed_solve(spaceconfig, h_impurity, chain)
    print('EG = %.4f'%res[0])
    pdb.set_trace()

def test_solve():
    from impurity import anderson_impurity
    from nrgmap.chain import Chain
    spaceconfig = SuperSpaceConfig([1,2,1])
    # Anderson Impurity
    U = 1.0
    h_impurity = anderson_impurity(U, U/2.)

    chain = Chain(tlist=[0.5*np.eye(2), 0.2*np.eye(2)] ,elist=[-0.2*np.eye(2), -0.5*np.eye(2)])
    res = ed_solve(spaceconfig, h_impurity, chain)
    mpo = build_mpo(spaceconfig, h_impurity, chain)
    for i, info in enumerate(solve_mpo(mpo)):
        print('E_%d = %.4f'%(i,info['E'].min()))
    print('EG = %.4f'%res[0])
    pdb.set_trace()

if __name__ == '__main__':
    test_solve()
