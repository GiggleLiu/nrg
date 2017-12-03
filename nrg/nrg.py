import numpy as np
from pymps.construct.opstringlib import opunit_cdag, opunit_c, opunit_N, opunit_Z, insert_Zs
from pymps.tensor import Tensor, BLabel
from pymps.tensor.zero_flux import btdot, is_zero_flux
from pymps.spaceconfig import SuperSpaceConfig
import pdb

from .build import BuildingSpace, build_mpo

def solve_mpo(mpo, max_keep=600, scaling_factor=1.0, shift_energy=True):
    '''
    solve a Wilson chain MPO using NRG iteration.

    Args:
        mpo (:obj:`MPO`): matrix product operator instance.
        max_keep (int, default=600): max number of states kept.
        scaling_factor (float, default=1.0): rescale the chain (MPO) by this factor after each iteration, usually taken as sqrt(Lambda).
        shift_energy (bool, default=True): if True, shift ground state energy is reset to 0 after each iteration.

    Returns:
        iterator: at each step, return a information dict with keys ['E', 'U', 'H', 'mask'].
    '''
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
        U, E, UD = H_.svd(2, cbond_str='u_%d'%(l+1), kernel='eigh', signs=[1,1,1,1], bmg=mpo.bmg)

        # truncation
        mask = E<=np.sort(E)[min(len(E), max_keep)-1]
        yield {'E':E, 'U':U, 'H':H_, 'mask':mask}
        U, UD = U.take(mask,-1), UD.take(mask,0)
        UD.labels[0] = UD.labels[0].chstr('d_%d'%(l+1))

        # update L to diagonal form, and rescale by a constant
        UH = btdot(U.conj(), H, sign_U, sign_H, mpo.bmg, check_bm=True)
        L = btdot(UH, UD.conj(), sign_UH, sign_UD, mpo.bmg, check_bm=True)
        if shift_energy:
            np.fill_diagonal(L[:,base_index,:], L[:,base_index,:].diagonal()-E.min())
        L = L*scaling_factor
        tensor_list = [U, UD, H, UH, L]
        sign_list = [sign_U, sign_UD, sign_H, sign_UH, sign_L]


def test_nrg_solve():
    from .impurity import anderson_impurity
    from nrgmap.chain import Chain
    spaceconfig = SuperSpaceConfig([1,2,1])
    # Anderson Impurity
    U = 1.0
    h_impurity = anderson_impurity(U, U/2.)

    chain = Chain(tlist=[0.5*np.eye(2), 0.2*np.eye(2)] ,elist=[-0.2*np.eye(2), -0.5*np.eye(2)])
    mpo = build_mpo(spaceconfig, h_impurity, chain)
    for i, info in enumerate(solve_mpo(mpo, shift_energy=False)):
        print('E_%d = %.4f'%(i,info['E'].min()))
    assert(abs(info['E'].min()+2.53369635456)<1e-6)

if __name__ == '__main__':
    test_nrg_solve()
