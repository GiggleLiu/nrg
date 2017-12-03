import numpy as np
from scipy.linalg import eigh, eigvalsh
from pymps.spaceconfig import SuperSpaceConfig
import scipy.sparse as sps
import pdb

from .build import BuildingSpace

def ed_solve(spaceconfig, h_impurity, chain, k=1):
    '''
    solve a Wilson Chain using exact diagonalization.

    Args:
        spaceconfig (SpaceConfig): Hilbert sapce configuration for a single site.
        h_impurity (2darray): hamiltonian on the impurity site, its dimension can be 0-1 sites.
        chain (:obj:`Chain`): Wilson chain generated in nrgmap.
        k (int): number of desired states.

    Returns:
        tuple(1darray, 2darray, sps.csr_matrix): energy, eigenvectors and hamiltonian.
    '''
    H, _update =  _ed_solve(spaceconfig, h_impurity, chain, k=1, step_wise=True)
    for ei, ti in zip(chain.elist, chain.tlist):
        H = _update(H, ei, ti)
    EG, VG = sps.linalg.eigsh(H, k=1, which='SA')
    return EG, VG, H

def ed_itersolve(spaceconfig, h_impurity, chain, k=1):
    '''
    solve a Wilson Chain using exact diagonalization, the iterative version.

    Args:
        see `ed_solve`.

    Returns:
        dict: information dict with keys ['E', 'H', 'U'].
    '''
    H, _update =  _ed_solve(spaceconfig, h_impurity, chain, k=1, step_wise=True)
    EG, VG = sps.linalg.eigsh(H, k=k, which='SA')
    yield {'E':EG,'H':H, 'U':VG}
    for ei, ti in zip(chain.elist, chain.tlist):
        H = _update(H, ei, ti)
        EG, VG = sps.linalg.eigsh(H, k=k, which='SA')
        yield {'E':EG,'H':H, 'U':VG}

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
    return H, _update


def test_ed_solve():
    from nrgmap.chain import Chain
    from .impurity import anderson_impurity
    spaceconfig = SuperSpaceConfig([1,2,1])
    # Anderson Impurity
    U = 1.0
    h_impurity = anderson_impurity(U, U/2.)

    chain = Chain(tlist=[0.5*np.eye(2), 0.2*np.eye(2)] ,elist=[-0.2*np.eye(2), -0.5*np.eye(2)])
    res = ed_solve(spaceconfig, h_impurity, chain)
    print('EG = %.4f'%res[0].item())
    assert(abs(res[0].item()+2.53369635456)<1e-6)

if __name__ == '__main__':
    test_ed_solve()
