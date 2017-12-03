from numpy import *
from scipy import sparse as sps
from scipy.linalg import eigh, eigvalsh
from scipy.sparse.linalg import eigsh
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
from matplotlib.pyplot import *
import pdb
import time

from tba.hgen import SpinSpaceConfig, E2DOS
from rglib.hexpand import RGHGen
from rglib.mps import OpUnitI, opunit_Sz, opunit_Sy, opunit_Sx, get_expect


class NIModel():
    '''
    Non-integrabal model.

    The Hamiltonian is: sum_{i=1 -> nsite}g*sx(i) + sum_{i=2 -> nsite-1}h*sz(i) + (h-J)(sz(1)+sz(nsite)) + sum_{i=1 -> nsite-1}J*sz(i)sz(i+1)

    Attributes:
        :J: number, exchange interaction at z direction.
        :g/h: number, the strength of transverse and longitudinal field.
    '''

    def __init__(self, g, h, nsite, J=1):
        self.spaceconfig = SpinSpaceConfig([2, 1])
        self.J, self.g, self.h, self.nsite = J, g, h, nsite
        scfg = self.spaceconfig
        I = OpUnitI(hndim=scfg.hndim)
        Sx = opunit_Sx(spaceconfig=scfg) * 2
        Sz = opunit_Sz(spaceconfig=scfg) * 2
        opc = 0
        for i in range(nsite):
            hi = h - J if i == 0 or i == nsite - 1 else h
            opc = opc + (hi * Sz.as_site(i) + g * Sx.as_site(i))
            if i != nsite - 1:
                opc = opc + J * Sz.as_site(i) * Sz.as_site(i + 1)
        self.opc = opc
        # mpo1=self.opc.toMPO(method='direct')
        # mpo2=self.opc.toMPO(method='addition')
        #mpo1.compress(); mpo2.compress()
        # self.mpo=mpo2

    def __str__(self):
        return 'NIModel: J=%s,g=%s,h=%s,nsite=%s' % (self.J, self.g, self.h, self.nsite)


def test_show_spec():
    nsites = [6, 8, 10]
    for nsite in nsites:
        E = get_spec(nsite, trunc_steps=[])
        wlist = linspace(-nsite * 2, nsite * 2, 200)
        dos = E2DOS(E, wlist, geta=0.4 / sqrt(len(E)) * nsite)
        ion()
        plot(wlist, dos)
    legend(nsites)
    pdb.set_trace()


def test_nn_project():
    '''Test for projection to nearest neighbor states.'''
    nsite = 10
    (E, mps), info = get_spec(nsite, trunc_steps=[], return_vecs=True)
    # for every vector, we will get <a(n-1)|b(m)> ~ <a(n)|b(m)>
    A = mps.get(9)
    ion()
    plot(abs(A[:, 0, 1000]))
    pdb.set_trace()


def test_loc_observe():
    nsite = 10
    (E, mps), info = get_spec(nsite, trunc_steps=[], return_vecs=True)
    model = info['model']
    sx = opunit_Sx(spaceconfig=model.spaceconfig)
    sz = opunit_Sz(spaceconfig=model.spaceconfig)
    sy = opunit_Sy(spaceconfig=model.spaceconfig)
    # measure Sx(i)
    ion()
    for isite in range(nsite):
        res = get_expect(sz.as_site(isite), mps)[0, 0].diagonal()
        # res=fft.fft(res)
        plot(res)
        pdb.set_trace()


def test_trunc_spec():
    nsite = 10
    E1 = get_spec(nsite, trunc_steps=[])
    E2 = get_spec(nsite, trunc_steps=[8])
    wlist = linspace(-nsite * 2, nsite * 2, 200)
    dos1 = E2DOS(E1, wlist, geta=0.4 / sqrt(len(E1)) * nsite) / len(E1)
    dos2 = E2DOS(E2, wlist, geta=0.4 / sqrt(len(E2)) * nsite) / len(E2)
    ion()
    plot(wlist, dos1)
    plot(wlist, dos2)
    legend('no-trunc', 'trunc')
    pdb.set_trace()


def get_spec(nsite, trunc_steps=[], return_vecs=False):
    '''
    Get the spectrum for a chain.

    Parameters:
        :nsite: int, number of sites.
        :trunc_steps: list, the steps to perform truncation.
        :return_vecs: bool,

    Return:
        E, info,
        (E, V), info
    '''
    g = (sqrt(5) + 5) / 8.
    h = (sqrt(5) + 1) / 4.
    model = NIModel(g=g, h=h, nsite=nsite)
    spaceconfig = model.spaceconfig
    hchain = model.opc
    expander = RGHGen(spaceconfig=spaceconfig,
                      hchain=hchain, evolutor_type='normal')

    H = sps.csr_matrix((1, 1), dtype='complex128')
    for i in range(nsite):
        H = expander.expand1(H)
        # stop, if not return vectors use eigvalsh
        if i == nsite - 1 and not return_vecs:
            E = eigvalsh(H.toarray())
            if i in trunc_steps:
                E = E[::2]
            return E, {'mode': model}
        E, U = eigh(H.toarray())
        if i in trunc_steps:
            E = E[::2]
            U = U[:, ::2]
        H = sps.diags(E)
        expander.trunc(U=sps.csr_matrix(U))
        # stop, if not vectors use eigvalsh
        if i == nsite - 1:
            return (E, expander.get_mps()), {'model': model}


# test_spec()
# test_trunc_spec()
# test_nn_project()
test_loc_observe()
