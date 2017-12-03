'''
Sample: Speudogap model with r = 1
'''
from builtins import input
from numpy import *
from matplotlib.pyplot import *
import time
import pdb

from nrgmap import quickmap, load_chain
from nrgmap.utils import get_wlist
from nrgmap.chainmapper import check_spec


def run():
    '''
    run this sample, visual check is quite slow!
    '''
    def rhofunc(w): return 0.5 / pi
    wlist = get_wlist(w0=1e-10, Nw=10000, mesh_type='log', Gap=0, D=1.)

    # create the discretized model
    chains = quickmap(wlist, rhofunc, Lambda=1.5,
                      nsite=60, nz=1, tick_type='log')

    plot_wlist = wlist
    docheck = input(
        'Check whether this chain recover the hybridization function?(y/n):') == 'y'
    if docheck:
        ion()
        cla()
        check_spec(rhofunc=rhofunc, chains=chains,
                   wlist=plot_wlist, smearing=0.2)
        print('Integrate should be %s, if being too small, oversmeared!' % (1. / pi))
        print('Press `c` to continue.')
        ylim(0, 0.2)

    dosave = input('Save the chain datas?(y/n):') == 'y'
    if dosave:
        for iz, chain in zip([1.0], chains):
            chain.save('data/flatband_%s' % iz)


def load():
    chain = load_chain('data/flatband_1.0')
    pdb.set_trace()


if __name__ == '__main__':
    # run()
    load()
