#/usr/bin/python
'''
Utilities for nrg.
'''
from numpy import *
from discretization import DiscHandler
from chainmapper import ChainMapper
from scale import EScale,load_scale,save_scale
from chain import save_chain,load_chain
from setting.local import RANK,SIZE,COMM
from matplotlib.pyplot import *
from nrg_setting import PRECISION,NW,NX,NX_CHECK_DISC,SMEARING_CHECK_DISC,NW_CHECK_DISC,SMEARING_CHECK_CHAIN,NW_CHECK_CHAIN
import pdb,time

def get_scale_chain(token,rhofunc,Lambda,N,D,Gap=0.,z=1.,tick_type='log',Nw=NW,w_mesh_type='log',\
        append=False,check_disc=False,check_trid=False,check_method='eval',autofix=-1e-8,r=1.):
    '''
    Test for mapping 2x2 rho function to discretized model.

    token:
        token for storing and fetching datas.
    rhofunc:
        hybridization function rho(w).
    Lambda:
        scaling factor.
    N:
        maximum scaling depth.
    D:
        band-width
    Gap:
        Gap range.
    z:
        twisting parameters.
    tick_type:
        tick type, `log`arithmic for default.
            * `log` -> logarithmic tick,
            * `sclog` -> logarithmic ticks suited for superconductor.
            * `adaptive` -> adaptive ticks.
            * `linear` -> linear ticks.
            * `adaptive_linear` -> linear adaptive ticks.
    Nw:
        number of discretization points for rho(w), the larger the better the slower ...
    append:
        load datas instead of generating one.
    check_disc:
        check and plot the hybridization after discretization if True.
    check_trid:
        check and plot the hybridization after tridiagonalization if True.
    check_method:
        * `eval`: check eigenvalues.
        * `pauli`: pauli decomposition(suited for 2D).
    r:
        adaptive ratio for adaptive ticks.
    '''
    if append:
        #scale=EScale(Ticklist,Lambda,z,Gap)
        nz=1 if ndim(z)==0 else len(z)
        scale=load_scale(token,Lambda,N,nz)
        chain=load_chain(token)
        #check for tridiagonalization
        if check_trid and RANK==0:
            mapper=DiscHandler(token,N=N,D=D,Gap=Gap)  #token,maximum scale index and band range,Gap.
            cmapper=ChainMapper(prec=PRECISION)
            cmapper.check_spec(chain,mapper,rhofunc)
        return scale,chain

    mapper=DiscHandler(token,N=N,D=D,Gap=Gap)  #token,Lambda,maximum scale index and band range,Gap.
    print 'Setting up hybridization function ...'
    w0=1e-6
    if w_mesh_type=='log':
        w0=Lambda**(-N)
    elif w_mesh_type=='sclog':
        w0=Lambda**(-2*N+1)
    mapper.set_rhofunc(rhofunc,Nw=Nw,autofix=autofix,w0=w0,w_mesh_type=w_mesh_type)    #a function of rho(w), number of ws for each branch.
    print 'Done.'

    #perform mapping and get functions of epsilon(x),E(x) and T(x) for positive and negative branches.
    #epsilon(x) -> function of discretization mesh points.
    #E(x)/T(x) -> function of representative energy and hopping terms.
    funcs=mapper.quick_map(tick_type=tick_type,Lambda=Lambda,Nx=NX,r=r) #tick type,number samples for integration over x.
    (ef_neg,Ef_neg,Tf_neg),(ef,Ef,Tf)=funcs

    #check for discretization.
    if check_disc and RANK==0:
        if check_method=='eval':
            check_disc=mapper.check_disc_eval
        else:
            check_disc=mapper.check_disc_pauli
        check_disc(rhofunc,Ef,Tf,ef,sgn=1,Nx=NX_CHECK_DISC,smearing=SMEARING_CHECK_DISC,Nw=NW_CHECK_DISC)
        check_disc(rhofunc,Ef_neg,Tf_neg,ef_neg,sgn=-1,Nx=NX_CHECK_DISC,smearing=SMEARING_CHECK_DISC,Nw=NW_CHECK_DISC)

    #get a discrete model
    #extract discrete set of models with output functions of quick_map, a DiscModel instance will be returned.
    scale,disc_model=mapper.get_scale_model(funcs,Lambda=Lambda,z=z,append=False)

    #Chain Mapper is a handler to map the DiscModel instance to a Chain model.
    cmapper=ChainMapper(prec=PRECISION)
    chain=cmapper.map(disc_model)

    #save the chain, you can get the chain afterwards by load_chain method or import it to other programs.
    #data saved:
    #   data/<token>.tl.dat -> representative coupling of i-th site to the previous site(including coupling with impurity site - t0), float view for complex numbers, shape is raveled to 1 column with length: nband x nband x nz x N(chain length) x 2(complex and real).
    #   data/<token>.el.dat -> representative energies, stored same as above.
    #   data/<token>.info.dat -> shape information, (Chain length,nz,nband,nband)
    if RANK==0:
        save_chain(token,chain)
        save_scale(token,scale)

    #check for tridiagonalization
    if check_trid and RANK==0:
        cmapper.check_spec(chain,mapper,rhofunc,Nw=NW_CHECK_CHAIN,smearing=SMEARING_CHECK_CHAIN,mode=check_method)

    return scale,chain


