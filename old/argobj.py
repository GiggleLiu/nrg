from scipy import *
from matplotlib.pyplot import *
from scipy.interpolate import splrep, splev, interp1d
from core.mathlib import log_gaussian, gaussian, lorenzian
from core.utils import matrix_spline, H2G, bcast_dot
from core.utils import kk_smooth as kk
from core.matrixlib import eigen_combine, eigh_sorted, eig_sorted
from rgoplib import rgop_c, rgop_n, rgop_c3, rgop_cd
from tdnrg import FDMManager, RDMManager
from scipy.sparse import issparse
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
from scipy.interpolate import *
from numpy.linalg import inv, eigh
from rgobj import RGRequirement
from binner import get_binner
from copy import deepcopy
from setting.local import COMM, SIZE, RANK
import pdb
import time
import warnings
import pickle


def get_expect_RDM(scale, dm, which, smearing_method='log_gaussian', b=0.4, diag=False):
    '''
    Get spectrum for zero temperature with whole Energy spectrum.

    scale:
        the scale.
    dm:
        an DMManager instance.
    which:
        `A`,`B` or both.
    smearing_method:
        method for smearing peaks.
        * `gaussian`(default): use gaussian peak.
        * `log_gaussian`: use log_gaussian peak(symmetric function).
        * `lorenzian`: use lorenzian peak.
    b:
        broadening
    diag:
        the spectrum function is diagonal if True(spin-z conservative).
    '''
    b = array([b] * len(x)).reshape(x.shape)
    if smearing_method == 'log_gaussian':
        smethod = log_gaussian
    elif smearing_method == 'gaussian':
        smethod = gaussian
    elif smearing_method == 'lorenzian':
        smethod = lorenzian
    else:
        warnings.warn('Unknow smearing method -set to default `gaussian`')
        smethod = gaussian

    # prepair data
    nband = self.nband  # the only supported case!
    label_up = 'f%s%sd' % (self.site, 'up')
    label_dn = 'f%s%sd' % (self.site, 'dn')
    label_up3 = 'fff%s%s' % (self.site, 'up')
    label_dn3 = 'fff%s%s' % (self.site, 'dn')
    ftrackers = [dm.H.trackers[label_up], dm.H.trackers[label_dn]]
    ffftrackers = [dm.H.trackers[label_up3], dm.H.trackers[label_dn3]]
    etracker = dm.Etracker
    bmtracker = dm.bmtracker
    needtrunc = etracker.beforetrunc
    plotmode = self.setting['plotmode']
    if plotmode == 'even':
        iterator = xrange(1, scale.N + 1, 2)
    elif plotmode == 'odd':
        iterator = xrange(0, scale.N + 1, 2)
    else:
        iterator = xrange(scale.N + 1)

    # get A(w),B(w)
    wlist = []
    Alist = []
    Blist = []
    # note that the i here stands for hamiltonian iteration(site index + 1).
    for i in iterator:
        t0 = time.time()
        fupd, fdnd = fds = ftrackers[0].get(
            i).tocsr(), ftrackers[1].get(i).tocsr()  # [fupd,fdnd]
        fffs = ffftrackers[0].get(i).tocsr(), ffftrackers[1].get(
            i).tocsr()  # [fupd,fdnd]
        rho = dm.rholist[i]
        rescalefactor = 1 if i == 0 else 1. / scale.scaling_factor[i - 1]
        Ei = etracker.get(i)
        if needtrunc:
            Ei = Ei[bmtracker.get(i).kpmask]
        # specify energy spectrum to extract
        wfactor = self.setting['wfactor'] * rescalefactor
        ws = array([wfactor * scale.D[0], wfactor * scale.D[1]])
        A = zeros([len(ws), nband, nband], dtype='complex128')
        B = zeros([len(ws), nband, nband], dtype='complex128')
        delta_peak = smethod(x=ws[..., newaxis, newaxis],
                             mean=Ei[:, newaxis] - Ei[newaxis, :], b=b)
        for spin1 in xrange(nband):
            F3 = fffs[spin1]
            M1 = fds[spin1]
            for spin2 in xrange(nband):
                if diag and spin2 != spin1:
                    continue
                M2 = fds[spin2]
                rM = (M2.T.tocsr().dot(rho.T)).T + M2.tocsr().dot(rho)
                for iw in xrange(len(ws)):
                    dp = delta_peak[iw]
                    # multiply dp
                    if issparse(rM):
                        rp = rM.multiply(dp)
                    else:
                        rp = rM * dp
                    A[iw, spin1, spin2] = (M1.conj().multiply(rp)).sum().real
                    B[iw, spin1, spin2] = (F3.T.multiply(rp)).sum().real
        wlist.append(ws)
        Alist.append(A)
        Blist.append(B)
        t1 = time.time()
        print rho.shape, '(elapse: %s)' % (t1 - t0)
    wlist = concatenate(wlist)
    Alist = concatenate(Alist, axis=0)
    Blist = concatenate(Blist, axis=0)
    rightorder = argsort(wlist)
    wlist = wlist[rightorder]
    Alist = Alist[rightorder]
    Blist = Blist[rightorder]
    return wlist[1:-1], Alist[1:-1], Blist[1:-1]


def get_expect_FDM(scale, dm, opspace, binner, diag=False):
    '''
    Get spectrum for zero temperature with Full-density matrix.

    scale:
        the scale.
    dm:
        an DMManager instance.
    opspace:
        a tuple of (lops,rops,H), left operator space and right operator space and hermion indicator(`T` for True and 'F' for False, e.g. 'TF' for performing hermion conjugate to left operator space).
    diag:
        the spectrum function is diagonal if True(spin-z conservative).

    *return*:
        wlist, Alist
    '''
    # prepair data and initalize dm.
    lops, rops, H = opspace
    nband = len(lops)
    for label in lops + rops:
        if not dm.ops.has_key(label):
            dm.init_op_list(label)
    blist = ndarray((nband, nband), dtype='O')
    for spin1 in xrange(nband):
        lop = lops[spin1]
        for spin2 in xrange(nband):
            if diag and spin2 != spin1:
                continue
            rop = rops[spin2]
            blist[spin1, spin2] = dm.get_spec(lop, rop, deepcopy(binner), H=H)
    return blist


class ARGobj(object):
    '''measurable objects measured after all RG iteration. (like DM-NRG)'''

    def __init__(self, label, requirements=None):
        self.label = label
        self.requirements = []
        if requirements != None:
            self.requirements = requirements
        self.setting = {}
        self.cache = {}
        self.data = None
        self.subtract_env = False
        self.time_dependant = True

    def __str__(self):
        return '<TD-NRG measurable object -> %s>' % (self.label)

    def save_data(self, filename):
        '''
        save data.
        '''
        f = open(filename, 'w')
        pickle.dump(self.data, f)
        f.close()

    def load_data(self, filename):
        '''
        load data.
        '''
        f = open(filename, 'r')
        self.data = pickle.load(f)
        f.close()

    def get_expect(self, dm):
        '''
        get expectation value from dm.
        '''
        raise Exception('Error', 'Not Implemented!')

    def measure(self, scale, tdmanager, **kwargs):
        '''
        measure an operator.

        scale:
            the scale instance.
        tdmanager:
            the timeline manager, and instance TDManager.
        '''
        mval = array([self.get_expect(scale, tdmanager.tls[i], **kwargs)
                      for i in xrange(scale.nz)])
        self.data = mval.mean(axis=0)

    def show(self):
        '''show data'''
        if self.data is None:
            warnings.warn(
                'No Data is available! You need to measure this observable before showing.')


class ARGobj_A(ARGobj):
    '''
    Spectrum function for specific site.
    a flow element is defined as [w,A(w),broadening]

    label:
        the label.
    spinindex/site:
        the specific site/spin index to measure.
    T:
        the temperature, default is 0
    wfactor:
        the relative w to measure with respect to the energy scale at each expansion.
    bfactor:
        broadening factor.
    '''

    def __init__(self, label='A', spinindex=0, site=0, T=0, *args, **kwargs):
        super(ARGobj_A, self).__init__(label)
        self.site = site
        self.T = T
        self.spinindex = spinindex if ndim(spinindex) == 1 else [spinindex]
        for spin in self.spinindex:
            rop = rgop_cd(site, spin)
            self.requirements.append(RGRequirement(
                rop.label, tp='op', islist=True, beforetrunc=True, info={'kernel': rop}))
        self.time_dependant = False

    def get_expect(self, scale, dm, diag=False, **kwargs):
        '''
        Get the expectation value of Tchi.

        dm:
            the hamiltonian instance.
        **kwargs:
            * FDM: wmin(-1),wmax(1),smearing_method(gaussian), b(0.5), nw(50)
            * RDM: smearing_method(log_gaussian), b(0.4)
        '''
        if isinstance(dm, FDMManager):
            fds = ['f%s%sd' % (self.site, 'up' if spin == 0 else 'dn')
                   for spin in self.spinindex]
            wlist, Alist = get_expect_FDM(scale, dm, opspace=(
                fds, fds, 'TF'), diag=diag, **kwargs)
            return wlist, Alist
        elif isinstance(dm, RDMManager):
            ms = get_expect_RDM(scale, dm, diag=diag, *args, **kwargs)
            return ms
        else:
            raise Exception(
                'parameter dm is not qualified @get_expect! should be a desity matrix NRG manager!')
        return ms

    def measure(self, scale, dmmanagers, threadz=False, **kwargs):
        '''
        measure an operator(overload).

        scale:
            the EScale instance.
        tdmanager:
            the timeline manager, and instance TDManager.
        threadz:
            multi-threading over z.
        **kwargs:
            * FDM: wmin(-1),wmax(1),smearing_method(gaussian), b(0.5), tol(0 if zero temperature else 1e-12), nw(50)
            * RDM: smearing_method(log_gaussian), b(0.4)
        '''
        AL = []
        ntask = (scale.nz - 1) / SIZE + 1
        for i in xrange(scale.nz):
            if threadz:
                if i / ntask == RANK:
                    print 'Measuring Sigma on core %s with Hamiltonian size %s.' % (RANK, dmmanagers[i].H.N)
                    w, A = self.get_expect(scale, dmmanagers[i], **kwargs)
                    AL.append(A)
            else:
                w, A = self.get_expect(scale, dmmanagers[i], **kwargs)
                AL.append(A)
        if threadz:
            AL = COMM.gather(AL, root=0)
            if RANK == 0:
                AL = concatenate(AL, axis=0)
                w, A = w, mean(AL, axis=0)
            A = COMM.bcast(A, root=0)
            self.data = w, A
        else:
            self.data = w, mean(AL, axis=0)

    def show(self, spin, *args):
        '''
        show spectrum of specific matrix element.
        '''
        A = self.data[1][:, spin[0], spin[1]]
        wlist = self.data[0]
        plts = []
        plts += plot(wlist, A.real)
        if spin[0] != spin[1]:
            plts += plot(wlist, A.imag)
            legend = [r'Im[A_{%s%s}]' % (
                spin[0], spin[1]), r'Im[A_{%s%s}]' % (spin[0], spin[1])]
        else:
            legend = [r'A_{%s%s}' % (spin[0], spin[1])]


class ARGobj_N(ARGobj):
    '''
    occupatian rate of specific site.

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

    def __init__(self, tlist, spinconfig='up', label='n', site=0, *args, **kwargs):
        super(ARGobj_N, self).__init__(label, requirements=None)
        self.tlist = tlist
        self.site = site
        rop = rgop_n(site, spinconfig)
        rop.label = self.label
        self.requirements.append(RGRequirement(
            rop.label, tp='op', islist=True, beforetrunc=True, info={'kernel': rop}))

    def get_expect(self, scale, dm):
        '''get the expectation value of <n>'''
        return array([[tml.get_expect('n', dt, f=f) for dt in self.tlist] for f in xrange(1, tml.nh)]).T

    def show(self):
        '''show data'''
        plot(self.tlist, self.data, lw=2)
        ylim(0., 1.)
        ylabel(self.label)


class ARGobj_Sigma(ARGobj):
    '''
    Self Energy function for specific site.
    a flow element is defined as [w,A(w),broadening]

    label:
        the label.
    wfactor:
        the relative w to measure with respect to the energy scale at each expansion.
    bfactor:
        broadening factor.
    '''

    def __init__(self, U, label='Sigma', site=0, nband=2, *args, **kwargs):
        self.U = U
        self.site = site
        self.nband = nband
        requirements = []
        for spin in [0, 1]:
            rop = rgop_cd(site, spin)
            rop3 = rgop_c3(site, spin)
            requirements.append(RGRequirement(
                rop.label, tp='op', islist=True, beforetrunc=True, info={'kernel': rop}))
            requirements.append(RGRequirement(
                rop3.label, tp='op', islist=True, beforetrunc=True, info={'kernel': rop3}))
        self.time_dependant = False
        super(ARGobj_Sigma, self).__init__(label, requirements)

    def get_sigma(self, wlist=None, interp='spline', kk_method='mac'):
        '''
        kk-relation to get (index1,index2) component of Sigma, we ignored pi factor here, for we will get U*FG^{-1}.

        For reference:
            * Initial version: J. Phys.: Condens. Matter 10.8365.
            * The matrix version: PRB 79.214518

        wlist:
            the frequency space.
        interp:
            interpolation method.

            * `spline`
            * `pchip`
        kk_method:
            method to calculate kk-relation(Appl. Spectrosc. 42.952).

            * `mac` -> using Maclaurin-method.
            * `fft` -> using Successive Double Fourier Transformation method.
        '''
        nband = self.nband
        print 'Interpolating A(w),B(w) to get self-energy Sigma(w).'
        t0 = time.time()
        wl, Alist, Blist = self.data
        if not wlist is None:
            if interp == 'pchip':
                interpolator = pchip_interpolate
            elif interp == 'spline':
                interpolator = matrix_spline
            else:
                raise Exception('Undefined interpolation method @get_sigma.')
            Alist = interpolator(wl, Alist, wlist)
            Blist = interpolator(wl, Blist, wlist)
        else:
            wlist = wl
        Gilist = -pi * Alist
        Filist = -pi * Blist
        if kk_method == 'fft':
            Grlist = kk_fft(Gilist, 'i2r', expand=1.5)
            Frlist = kk_fft(Filist, 'i2r', expand=1.5)
        elif kk_method == 'mac':
            Grlist = kk(Gilist, 'i2r', wlist=wlist)
            Frlist = kk(Filist, 'i2r', wlist=wlist)
        else:
            raise Exception('Undefined calculation method for kk-relation!')
        #Slist=self.U*array([(Br+1j*Bi).dot(inv(Ar+1j*Ai)) for Ar,Ai,Br,Bi in zip(Arlist,Alist,Brlist,Blist)])
        # Srlist=(swapaxes(Slist.conj(),1,2)+Slist)/2.
        # Silist=(swapaxes(Slist.conj(),1,2)-Slist)*(-1j/2.)
        G = Grlist + 1j * Gilist
        F = Frlist + 1j * Filist
        # why here is a minus sign?!
        S = -self.U * bcast_dot(F, inv(G))
        t1 = time.time()
        print 'Elapse -> %s' % (t1 - t0)
        return S, G, F

    def get_sigma_eig(self, wlist=None, interp='spline', kk_method='mac'):
        '''
        kk-relation to get (index1,index2) component of Sigma, we ignored pi factor here, for we will get U*FG^{-1}.

        For reference:
            * Initial version: J. Phys.: Condens. Matter 10.8365.
            * The matrix version: PRB 79.214518

        wlist:
            the frequency space.
        interp:
            interpolation method.

            * `spline`
            * `pchip`
        kk_method:
            method to calculate kk-relation(Appl. Spectrosc. 42.952).

            * `mac` -> using Maclaurin-method.
            * `fft` -> using Successive Double Fourier Transformation method.
        '''
        nband = self.nband
        print 'Interpolating A(w),B(w) to get self-energy Sigma(w).'
        t0 = time.time()
        wl, Alist, Blist = self.data
        if not wlist is None:
            if interp == 'pchip':
                interpolator = pchip_interpolate
            elif interp == 'spline':
                interpolator = matrix_spline
            else:
                raise Exception('Undefined interpolation method @get_sigma.')
            Alist = interpolator(wl, Alist, wlist)
            Blist = interpolator(wl, Blist, wlist)
        else:
            wlist = wl
        Gilist = -pi * Alist
        Filist = -pi * Blist
        Gievals, Gievecs = eigh_sorted(Gilist)
        # Fievals,Fievecs=eig_sorted(Filist)
        if kk_method == 'fft':
            Grevals = kk_fft(Gievals, 'i2r', expand=1.5)
            Frlist = kk_fft(Filist, 'i2r', expand=1.5)
        elif kk_method == 'mac':
            Grevals = kk(Gievals, 'i2r', wlist=wlist)
            Frlist = kk(Filist, 'i2r', wlist=wlist)
        else:
            raise Exception('Undefined calculation method for kk-relation!')
        #Slist=self.U*array([(Br+1j*Bi).dot(inv(Ar+1j*Ai)) for Ar,Ai,Br,Bi in zip(Arlist,Alist,Brlist,Blist)])
        # Srlist=(swapaxes(Slist.conj(),1,2)+Slist)/2.
        # Silist=(swapaxes(Slist.conj(),1,2)-Slist)*(-1j/2.)
        Gevals = Grevals + 1j * Gievals
        # Fevals=Frevals+1j*Fievals
        invG = eigen_combine(1. / Gevals, Gievecs)
        F = Frlist + 1j * Filist
        # why here is a minus sign?!
        S = -self.U * bcast_dot(F, invG)
        t1 = time.time()
        print 'Elapse -> %s' % (t1 - t0)
        return S, invG, F

    def get_sigma_naive(self, chain, recursive=False):
        '''
        get the naive self energy by `G0^{-1} - G^{-1}`.

        chain:
            the chain instance.
        recursive:
            the recursive approach.
        '''
        print 'Getting self-energy using Naive Method.'
        t0 = time.time()
        wlist = self.data[0]
        if recursive:
            G0 = chain.get_G0(wlist, geta=0.01)
        else:
            HL = chain.H0
            G0 = mean([H2G(w=wlist[:, newaxis, newaxis], h=H0,
                           geta=1e-2)[:, :2, :2] for H0 in HL], axis=0)
        Alist = self.data[1]
        Arlist = kk(Alist, 'i2r', wlist=wlist)
        Gr = -pi * (Arlist + 1j * Alist)
        sigma = inv(G0) - inv(Gr)
        t1 = time.time()
        print 'Elapse -> %s' % (t1 - t0)
        return sigma
        # G0=Gwmesh(G02,spaceconfig=scimp.spaceconfig,wlist=gwl,tp='r',geta=None)

    @staticmethod
    def check_sumrule(wlist, Alist, wspan=None):
        '''
        check for sumrule.

        wlist/wpan:
            a list of w/the region to perform integration.
        Alist:
            a list of A.
        '''
        if not wspan is None:
            wmask = (wlist < wspan[1]) & (wlist > wspan[0])
            wlist = wlist[wmask]
            Alist = Alist[wmask]
        print 'Check for Sum Rule:', trapz(Alist, wlist, axis=0)

    def show(self, viewpoint, wlist=None, interp='spline', *args):
        '''
        Show Imaginary and Real part of self energy.

        viewpoint:
            the matrix element.
        wlist/interp:
            the frequency space/interpolation method, use original one as default.
        '''
        plts = []
        nband = self.nband
        if wlist is None:
            wlist = self.data[0]
        Slist, Glist, Flist = self.get_sigma(interp=interp, wlist=wlist)
        i, j = viewpoint
        plts += plot(wlist, Slist[:, i, j].imag, lw=2)
        plts += plot(wlist, Glist[:, i, j].imag, lw=1)
        plts += plot(wlist, Flist[:, i, j].imag, lw=1)
        legends = [r'Im[$\Sigma_{%s%s}$]' % (i, j), r'Im$[G_{%s%s}(\omega)]$' % (
            i, j), r'Im$[F_{%s%s}(\omega)]$' % (i, j)]
        if i != j:
            plts += plot(wlist, Glist[:, i, j].real, lw=1, ls='--')
            plts += plot(wlist, Flist[:, i, j].real, lw=1, ls='--')
            legends += [r'Re$[G_{%s%s}(\omega)]$' %
                        (i, j), r'Re$[F_{%s%s}(\omega)]$' % (i, j)]
        legend(plts, legends, ncol=2)

    def get_expect(self, scale, dm, wspan, nw, diag=False, smearing_method='gaussian', b=1., w0=None, b0=None, **kwargs):
        '''
        Get the expectation value of Tchi.

        dm:
            the hamiltonian instance.
        wspan,nw:
            the frequency space.
        smearing_method,b:
            smearing method and broadening factor.
        w0:
            the critical frequency of this system below which, linear mesh and gaussian broadening will be used.
        **kwargs:
            * FDM: 
            * RDM: 
        '''
        if isinstance(dm, FDMManager):
            # binner is used to collect energies and generate spectrum.
            binner = get_binner(D=wspan, N=nw, w0=w0, tp='mixed')
            fds = ['f%s%sd' % (self.site, 'up'), 'f%s%sd' % (self.site, 'dn')]
            fffs = ['fff%s%s' % (self.site, 'up'), 'fff%s%s' %
                    (self.site, 'dn')]
            binner_A = get_expect_FDM(scale, dm, opspace=(
                fds, fds, 'TF'), diag=diag, binner=deepcopy(binner), **kwargs)
            binner_B = get_expect_FDM(scale, dm, opspace=(
                fffs, fds, 'FF'), diag=diag, binner=deepcopy(binner), **kwargs)
            wlist = binner.bins
            nband = len(fds)
            Alist = ndarray([len(wlist), nband, nband], dtype='complex128')
            Blist = ndarray([len(wlist), nband, nband], dtype='complex128')
            for i in xrange(nband):
                for j in xrange(nband):
                    Alist[:, i, j] = binner_A[i, j].get_spec(
                        wlist=wlist, smearing_method=smearing_method, b=b * 50, b0=b0)
                    Blist[:, i, j] = binner_B[i, j].get_spec(
                        wlist=wlist, smearing_method=smearing_method, b=b * 50, b0=b0)
            return wlist, Alist, Blist
        elif isinstance(dm, RDMManager):
            ms = self.__get_expect_RDM__(scale, dm, diag=diag, *args, **kwargs)
            return ms
        else:
            raise Exception(
                'parameter dm is not qualified @get_expect! should be a desity matrix NRG manager!')

    def measure(self, scale, dmmanagers, threadz=False, **kwargs):
        '''
        measure an operator(overload).

        scale:
            the EScale instance.
        tdmanager:
            the timeline manager, and instance TDManager.
        threadz:
            multi-threading over z.
        **kwargs:
            * FDM: wmin(-1),wmax(1),smearing_method(gaussian), b(0.5), tol(0 if zero temperature else 1e-12), nw(50)
            * RDM: smearing_method(log_gaussian), b(0.4)
        '''
        AL = []
        BL = []
        ntask = (scale.nz - 1) / SIZE + 1
        for i in xrange(scale.nz):
            if threadz:
                if i / ntask == RANK:
                    print 'Measuring Sigma on core %s with Hamiltonian size %s.' % (RANK, dmmanagers[i].H.N)
                    w, A, B = self.get_expect(scale, dmmanagers[i], **kwargs)
                    AL.append(A)
                    BL.append(B)
            else:
                w, A, B = self.get_expect(scale, dmmanagers[i], **kwargs)
                AL.append(A)
                BL.append(B)
        if threadz:
            AL = COMM.gather(AL, root=0)
            BL = COMM.gather(BL, root=0)
            if RANK == 0:
                AL = concatenate(AL, axis=0)
                BL = concatenate(BL, axis=0)
                w, A, B = w, mean(AL, axis=0), mean(BL, axis=0)
            A, B = COMM.bcast(A, root=0), COMM.bcast(B, root=0)
        else:
            A, B = mean(AL, axis=0), mean(BL, axis=0)
        print 'Fixing Occational Negative Eigenvalues of A.'
        aevals, aevecs = eigh(A)
        amin = aevals.min()
        if amin < 0:
            print 'Fixing Negative spectrum up to %s' % amin
            aevals[aevals < 0] = 0
        Alist = eigen_combine(aevals, aevecs)
        self.data = w, A, B

    def save_data(self, filename):
        '''
        save data.
        '''
        f = open(filename, 'w')
        pickle.dump(self.data, f)
        f.close()

    def load_data(self, filename, fix_neg=False):
        '''
        load data.

        filename:
            the filename.
        fix_neg:
            fix negative eigenvalues of A of True.
        '''
        f = open(filename, 'r')
        self.data = pickle.load(f)
        if fix_neg:
            print 'Fixing Occational Negative Eigenvalues of A.'
            Alist = self.data[1]
            aevals, aevecs = eigh(Alist)
            amin = aevals.min()
            if amin < 0:
                print 'Fixing Negative spectrum up to %s' % amin
                aevals[aevals < 0] = 0
            self.data = self.data[0], eigen_combine(
                aevals, aevecs), self.data[2]
        f.close()
