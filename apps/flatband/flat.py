import numpy as np
import pdb
import sys
from profilehooks import profile
sys.path.insert(0, '.')
from nrg import nrg, measure, build, ed
from nrg.baths import flatband
from nrg.impurity import anderson_impurity, empty_impurity
from pymps.spaceconfig import SuperSpaceConfig
import matplotlib.pyplot as plt

def run_test():
    '''comparing NRG with ED.'''
    spaceconfig = SuperSpaceConfig([1,2,1])
    # flat band chain.
    chain = flatband.gen_chain()
    chain.elist = chain.elist*np.eye(2)
    chain.tlist = chain.tlist*np.eye(2)
    # Anderson Impurity
    U = 1.0
    h_impurity = anderson_impurity(U, U/2.)

    mpo = build.build_mpo(spaceconfig, h_impurity, chain)
    max_keep = 400
    s1, s2 = nrg.solve_mpo(mpo, max_keep=max_keep, shift_energy=False), nrg.solve_mpo(mpo,max_keep=max_keep, shift_energy=False)
    s0 = ed.ed_itersolve(spaceconfig, h_impurity, chain)
    info0 = s0.__next__()
    info1 = s1.__next__()
    print('Step %d, EG = %.4f, %d states kept'%(0,info1['E'].min(),info1['mask'].sum()))
    print('  Exact = %.4f'%info0['E'][0])
    for i, (info0, info1, info2) in enumerate(zip(s0, s1,s2)):
        print('Step %d, EG = %.4f, %d states kept'%(i+1,info1['E'].min(),info1['mask'].sum()))
        print('  Exact = %.4f'%info0['E'][0])

def run_nrg():
    '''
    repeat results in paper
        PRB.21.1003,
        Krishna-murthy, H., Wilkins, J., & Wilson, K. (1980).
        Fig 9, line A.
    '''
    # setup parameters
    beta0 = 0.5                 # teperature factor, see RMP paper.
    quantum_numbers = 'QM'      # use U(1) good quantum number charge and spin.
    U = 1e-2                    # Hubbard interation strength.
    max_keep = 600              # keep 600 states.
    Lambda = 2.5                # discretization scaling facter, corresponding energy scaling factor is sqrt(2.5)
    spaceconfig = SuperSpaceConfig([1,2,1])  # 1 site x 2 spin x 1 orbital

    # generate wide-band metal bath.
    chain = flatband.gen_chain(Lambda=Lambda, nsite=70, G=U/12.66/np.pi)
    chain.elist = chain.elist*np.eye(2)
    chain.tlist = chain.tlist*np.eye(2)

    # Anderson Impurity
    h_impurity = anderson_impurity(U, U/2.)

    # build two chains, mpo with impurity, mpo2 without impurity.
    # calling solve_mpo will return a iterator, here we get s1 and s2.
    mpo = build.build_mpo(spaceconfig, h_impurity, chain, quantum_numbers=quantum_numbers)
    s1 = nrg.solve_mpo(mpo, max_keep=max_keep, scaling_factor=np.sqrt(Lambda))
    mpo2 = build.build_mpo(spaceconfig, empty_impurity(), chain, quantum_numbers=quantum_numbers)
    s2 = nrg.solve_mpo(mpo2, max_keep=max_keep, scaling_factor=np.sqrt(Lambda))

    # running the first step, and fix the scaling of empty chain.
    info1 = s1.__next__()
    mpo2.OL[0]*=np.sqrt(Lambda)
    print('Step %d, EG = %.4f, %d states kept'%(0,info1['E'].min(),info1['mask'].sum()))

    # plot setting
    plt.ion()
    plt.figure(figsize=(6,4))
    plt.title('Tchi flow')
    plt.ylabel(r'$T\chi_{\rm imp}$')
    plt.xlabel('step')
    plt.ylim(0,0.25)
    pltobj = plt.plot([0],[measure.tchi(quantum_numbers=quantum_numbers, beta=beta0, **info1)])[0]

    def _update_plot(data):
        x, y = pltobj.get_data()
        pltobj.set_data(np.append(x, [x[-1]+1]), np.append(y, [data]))
        plt.draw()
        plt.xlim(0,x[-1]+1)
        plt.pause(0.01)

    # run remaining steps.
    for i, (info1, info2) in enumerate(zip(s1,s2)):
        print('Step %d, EG = %.4f, %d states kept'%(i+1,info1['E'].min(),info1['mask'].sum()))

        # plot T*chi, impurity T*chi is defined as the difference of T*chi between chians with-without impurity.
        tchi = measure.tchi(quantum_numbers = quantum_numbers, beta=beta0, **info1)
        tchi0 = measure.tchi(quantum_numbers = quantum_numbers, beta=beta0, **info2)
        _update_plot(tchi-tchi0)

    pdb.set_trace()
    plt.savefig('Tchiflow.pdf')

if __name__ == '__main__':
    # run_test()
    run_nrg()
