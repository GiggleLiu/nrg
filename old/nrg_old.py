from scipy import *
from matplotlib.pyplot import *
from scipy import sparse as sps
import pdb
import time

from pymps.blockmarker import eigbh, trunc_bm
from .impuritymodel import ImpurityModel, scale_bath
from .plotlib import plot_spectrum

__all__ = ['NRGSolve', 'callback_spec']


def NRGSolve(impurity, bath, Lambda, bmg=None, maxN=800, call_back=None):
    '''
    Using NRG iteration method to solve a chain.

    Parameters:
        :impurity: <Impurity>,
        :bath: <Chain>,
        :Lambda: num/1d array, the scaling factor, start from the impurity spin.
        :bmg: <BlockMarkerGenerator>,
        :maxN: integer, the maximum retained energy levels.
        :call_back: function, f(iiter,E,kpmask,expander,bm,...) called after each iteraction.

    Return:
        dict of (EL, expander, bms), relative energy, expander, block markers for each iteraction.

        Note, elist is rescaled back.
    '''
    # scale the bath.
    if ndim(Lambda) == 0:
        Lambda = append([1], Lambda**arange(bath.nsite))
    bath = scale_bath(bath, Lambda=Lambda[1:])
    # generate the new model.
    model = ImpurityModel(impurity, bath)
    if impurity.H0 is None:
        Lambda = Lambda[1:]  # run without impurity
    nsite = model.nsite  # bath.nsite+1 if impurity is not Null

    # construct expander and blockmaker generator.
    expander = ExpandGenerator(model.get_opc(), evolutor_type='masked')

    # ready to run dmrg iteration.
    elist = []
    H = sps.csr_matrix((1, 1), dtype='complex128')
    if bmg is not None:
        bm = bmg.bm0
        bms = [bm]
    for i in range(nsite):
        print('Running iteraction %s' % (i + 1))
        # rescale H and add one site,
        if i != 0:
            H *= Lambda[i] / Lambda[i - 1]
        H = expander.expand1(H)
        if bmg is not None:
            bm, pm = bmg.update1(bm, compact_form=True)
            # block diagonalize hamiltonian and get eigenvalues.
            H_bd = H[pm][:, pm]
            if not bm.check_blockdiag(H_bd):
                raise Exception(
                    'Hamiltonian is not block diagonal with good quantum number %s' % good_number)
            E, U = eigbh(H_bd, bm=bm)
            # mul permutaion matrix to U to get a true one.
            U = U.tocsr()[argsort(pm)]
        else:
            E, U = eigh(H)

        # perform the truncation
        E_sorted = sort(E)
        kpmask = E <= E_sorted[min(maxN, len(E_sorted)) - 1]
        E = E[kpmask]
        if bmg is not None:
            bm = trunc_bm(bm, kpmask)
            bms.append(bm)
        expander.trunc(U=U, kpmask=kpmask)

        # rescale and construct the hamiltonian, scale back the energies.
        H = sps.diags(E)
        elist.append((E - E.min()) / Lambda[i])

        # call back.
        if call_back is not None:
            call_back(E=E, expander=expander, kpmask=kpmask,
                      iiter=i, bm=None if bmg is None else bm)
    if bmg is None:
        return {'EL': elist, 'expander': expander, 'model': model}
    else:
        return {'EL': elist, 'expander': expander, 'bms': bms, 'model': model}


def callback_spec(NE=20, target_block=None, **kwargs):
    '''
    Show spectrum after each iteration.

    Paramters:
        :NE: int, maximum number of levels shown.
        :target_block: 1d array/function, block label or f(iiter) as block label.
    '''
    E = kwargs['E']
    iiter = kwargs['iiter']
    bm = kwargs['bm']
    if target_block is None or bm is None:
        E_sorted = sort(E)
        plot_spectrum(E_sorted[:NE] - E_sorted[0], offset=[iiter, 0.], lw=1)
    else:
        if hasattr(target_block, '__call__'):
            target_block = target_block(iiter)
        E = bm.extract_block(E, (target_block,), axes=(0,),
                             uselabel=True) - E.min()
        E_sorted = sort(E)
        plot_spectrum(E_sorted[:NE] - E_sorted[0], offset=[iiter, 0.], lw=1)
    gcf().canvas.draw()
    pause(0.01)
