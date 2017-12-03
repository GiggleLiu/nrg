'''
Sample: Speudogap model with r = 1
'''
from builtins import input
from numpy import *
from matplotlib.pyplot import *
import pdb, os

from nrgmap import quickmap, load_chain
from nrgmap.utils import get_wlist
from nrgmap.chainmapper import check_spec


def gen_chain(Lambda=1.5, nsite=40, G=0.5, dosave=False, docheck=False):
    '''
    run this sample, visual check is quite slow!
    '''
    folder = os.path.join(os.path.dirname(__file__), 'data')
    rhofunc=lambda w: G/pi
    wlist=get_wlist(w0=1e-10,Nw=10000,mesh_type='log',Gap=0,D=1.)

    #create the discretized model
    chains=quickmap(wlist,rhofunc,Lambda=Lambda,nsite=nsite,nz=1,tick_type='log')

    if docheck:
        plot_wlist=wlist
        ion();cla()
        check_spec(rhofunc=rhofunc,chains=chains,wlist=plot_wlist,smearing=0.2)
        print('Integrate should be %s, if being too small, oversmeared!'%(1./pi))
        print('Press `c` to continue.')
        ylim(0,0.2)

    if dosave:
        for iz,chain in zip([1.0],chains):
            chain.save(os.path.join(folder, 'flatband_%s'%iz))
    return chains[0]


def load():
    folder = os.path.join(os.path.dirname(__file__), 'data')
    chain = load_chain(os.path.join(folder, 'flatband_1.0'))
    return chain

if __name__=='__main__':
    gen_chain()
