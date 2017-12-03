import numpy as np
import pdb

def tchi(E, U, mask, quantum_numbers, beta, **kwargs):
    '''
    T*chi = <S_z^2> - <S_z>^2
    '''
    index = quantum_numbers.index('M')
    Sz = U.labels[-1].bm.inflate().qns[:,index]/2.
    pM = np.exp(-beta*E)
    pM = pM/pM.sum()
    m_Sz = (pM*Sz).sum()
    m_Sz2 = (pM*Sz**2).sum()
    return m_Sz2-m_Sz**2
