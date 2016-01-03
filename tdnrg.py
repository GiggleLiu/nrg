from scipy import *
from matplotlib.pyplot import *
from scipy import sparse as sps

from mathlib import lorenzian,gaussian,log_gaussian,log_gaussian_fast,log_gaussian_var
from binner import get_binner
import pdb,time

class DMManager(object):
    '''
    Manager class for full density matrix.
    '''
    def __init__(self,evolutor,elist):
        self.evolutor=evolutor
        self.elist=elist
        self.T=T
        self.rholist=[]
        self.ops={}

    def get_rho0(self,m,degeneracy_eps=1e-10,which='all'):
        '''
        get density matrix of m-th iteration.

        m:
            the iteration index.
        degeneracy_eps:
            distance of two energy levels to view as degenerate at 0 temperature, 1e-10 by default.
        which:
            which part of rho, 'kp'/'dis'/'all'(default)
        '''
        T=self.T
        E=self.elist[m]
        kpmask=self.evolutor.kpmask(m)
        E=E-E.min()
        if T==0:
            rho=(E<degeneracy_eps).astype('complex128')
        else:
            rho=exp(-E/T)
        if which=='kp':
            if etracker!=None and etracker.beforetrunc:
                rho=rho[kpmask]
        elif which=='dis':
            dismask=~kpmask
            rho=rho[dismask]
        Z=sum(rho)
        rho/=Z
        return sps.diags(rho,0)

class FDMManager(DMManager):
    '''
    Full Density Matrix - NRG
    reference: PRL 99.076402
    '''
    def __init__(self,H,T,degeneracy_eps=1e-10):
        super(FDMManager,self).__init__(H=H,T=T)
        self.wnlist=[]
        self.init_rho_list(degeneracy_eps)

    def get_rho_serie(self,n,degeneracy_eps=1e-10):
        '''
        get a serie of rho(m,n) for m<n with n fixed.

        n:
            the fixed shell index.
        degeneracy:
            the degeneracy energy.
        '''
        T=self.T
        rholist=[]
        N=self.H.N
        for m in xrange(n,-1,-1):
            if m==n:
                #get the DD component
                kpmask=self.evolutor.kpmask(m)
                E=self.elist[m]
                E=E-E.min()
                if T==0:
                    rho=complex128(E<degeneracy_eps)
                else:
                    rho=exp(-E/T)
                if n!=N-1:
                    rho=rho[~kpmask]
                Zn=rho.sum()
                print 'Get Z=%s, E = %s ...'%(Zn,sort(E)[:4])
                if Zn!=0:
                    rho=rho/Zn
                if len(rho)==0:
                    rholist.append(sps.csr_matrix((0,0)))
                else:
                    rholist.append(sps.diags(rho,0))
            else:
                #infomation of last iteration
                rho0=rholist[-1]
                mask=self.evolutor.kpmask(m+1)
                if n!=N-1 and m==n-1:
                    mask=~mask
                A=self.evolutor.A(m+1)
                rl=[]
                for i in xrange(len(A)):
                    Ai=A[i].tocsc()[:,mask].tocsr()
                    rl.append(Ai.dot(rho0.tocsc()).dot(Ai.T.conj()))
                rholist.append(sum(rl,axis=0))
        rholist.reverse()
        return rholist

    def init_rho_list(self,degeneracy_eps=1e-15):
        '''
        initialize full density matrix.

        degeneracy_eps:
            the degeneracy energy detection.
        '''
        print 'Initializing mesh of rho(mn) ...'
        N=self.H.N
        hndim=self.H.spaceconfig.hndim
        rlist=[]
        wnlist=[]
        Znlist=[]
        for n in xrange(N):
            t0=time.time()
            if self.T==0 and n!=N-1:
                #only n=N-1 would be non-vanishing for T=0.
                rlist.append(None)
                wnlist.append(0)
                Znlist.append(0)
            else:
                rlist.append(self.get_rho_serie(n,degeneracy_eps=degeneracy_eps))
                #get wn
                Zn=rlist[n][n].diagonal().sum()
                Znlist.append(Zn)
            t1=time.time()
            print 'n = %s, Elapse -> %s'%(n,t1-t0)
        wnlist=array([float64(hndim)**(N-i-1) for i in xrange(N)])
        self.wnlist=wnlist/sum(wnlist*array(Znlist))
        self.rholist=[[rhonm*wn for rhonm in rhon] if rhon!=None else None for rhon,wn in zip(rlist,self.wnlist)]

    def init_op_list(self,label):
        '''
        get the operator list in each shell.

        label:
            the label of target operator.
        '''
        N=self.H.N
        optracker=self.H.trackers.get(label)
        bmtracker=self.bmtracker
        oplist=[]
        for n in xrange(N):
            kpmask=self.evolutor.kpmask(n)
            dsmask=~kpmask
            if n==N-1:
                kpmask,dsmask=dsmask,kpmask
            op=optracker.get(n).tocsr()
            #break into parts: KK, KD, DK, DD
            opk,opd=op[kpmask].tocsc(),op[dsmask].tocsc()
            oplist.append([opk[:,kpmask],opk[:,dsmask],opd[:,kpmask],opd[:,dsmask]])
        self.ops[label]=oplist

    def get_spec(self,B,C,binner,H):
        '''
        get the spectrum function of <<B|C>>.

        Parameters:
            :B/C: <OpUnit>, two operators.
            :binner: the binner instance to store data.
            :H: `FF`/`FT`/`TF`/`TT` -> hermion conjugate if `T`, the first and second character indicate B C respectively.

        Return:
            A tuple of -> (logarithmic frequency mesh, a specturm defined on this mesh)
        '''
        print 'Calculating Spectrum of <<%s(t)|%s>>...'%(B if H[0]=='F' else B+'.H',C if H[1]=='F' else C+'.H')
        N=self.H.N
        EN=self.elist[N-1]

        rholist=self.rholist   #indexing -> [n][m]
        if rholist==[] or B is None or C is None:
            raise Exception('Data not prepaired! Please intialize rho and operators!')
        dps=[]
        AL=[]
        t00=time.time()
        for n in xrange(N):
            #only n=N-1 would be non-vanishing.
            if self.T==0 and n!=N-1:
                continue
            rl=rholist[n]
            for m in xrange(n,-1,-1):
                B=self.evolutor.opat(B,m)
                C=self.evolutor.opat(C,m)
                em=self.Elist[m]
                kpmask=self.evolutor.kpmask(m)
                EK,ED=em[kpmask],em[~kpmask]
                rhomn=rl[m].tocsc()
                WL=[];EL=[]
                if m==n:
                    #for m==n, only DD component of rhomn is non-vanishing.
                    if H[1]=='F':
                        CKD,CDK,CDD=cm[1],cm[2],cm[3]
                    else:
                        CKD,CDK,CDD=cm[2].T.conj(),cm[1].T.conj(),cm[3].T.conj()
                    if H[0]=='F':
                        BKD,BDK,BDD=bm[1],bm[2],bm[3]
                    else:
                        BKD,BDK,BDD=bm[2].T.conj(),bm[1].T.conj(),bm[3].T.conj()
                    if n==N-1:
                        EK,ED=ED,EK
                    M11=CKD.tocsr().dot(rhomn).multiply(BDK.T)
                    M12=rhomn.tocsr().dot(CDK.tocsc()).multiply(BKD.T)
                    M2=(CDD.tocsr().dot(rhomn)+rhomn.tocsr().dot(CDD.tocsc())).multiply(BDD.T)
                    inds11=M11.nonzero()
                    inds12=M12.nonzero()
                    inds2=M2.nonzero()
                    if len(inds11[0])>0:
                        WL.append(array(M11[inds11])[0])
                        EL.append(EK[inds11[0]]-ED[inds11[1]])
                    if len(inds12[0])>0:
                        WL.append(array(M12[inds12])[0])  #weights
                        EL.append(ED[inds12[0]]-EK[inds12[1]]) #mean val
                    if len(inds2[0])>0:
                        WL.append(array(M2[inds2])[0])
                        EL.append(ED[inds2[0]]-ED[inds2[1]])
                else:
                    #for m<n, only KK component of rhonm is non-vanishing.
                    if H[1]=='F':
                        CKD,CDK=cm[1],cm[2]
                    else:
                        CKD,CDK=cm[2].T.conj(),cm[1].T.conj()
                    if H[0]=='F':
                        BKD,BDK=bm[1],bm[2]
                    else:
                        BKD,BDK=bm[2].T.conj(),bm[1].T.conj()
                    M11=CDK.tocsr().dot(rhomn).multiply(BKD.T)
                    M12=rhomn.tocsr().dot(CKD.tocsc()).multiply(BDK.T)
                    inds11=M11.nonzero()
                    inds12=M12.nonzero()
                    if len(inds11[0])>0:
                        WL.append(array(M11[inds11])[0])
                        EL.append(ED[inds11[0]]-EK[inds11[1]])
                    if len(inds12[0])>0:
                        WL.append(array(M12[inds12])[0])  #weights
                        EL.append(EK[inds12[0]]-ED[inds12[1]]) #mean val
                if len(EL)>0:
                    E=concatenate(EL)
                    W=concatenate(WL)
                else:
                    continue
                binner.push(el=E,wl=W)
        t1=time.time()
        print 'Elapse -> %s'%(t1-t00)
        return binner

class RGTimeLine(object):
    '''
    Time dependant RG hamiltonian.

    hlist:
        a hamiltonian prototype.
    tlist:
        a list of time for each hamiltonian.
    '''
    def __init__(self,dmlist,tlist):
        self.tlist=tlist
        self.dmlist=dmlist
        self.S_list={}

    def init_rho_list(self,i):
        '''initialize rho list of i-th time.'''
        print 'initializing rho for time slice ->',i
        t0=time.time()
        dm=self.dmlist[i]
        dm.init_rho_list()
        t1=time.time()
        print 'done(time cost: %s)'%(t1-t0,)

    def init_S_list(self,f,i=0,mode='ft'):
        '''
        initialize projection matrix connecting f and i.

        f:
            index of final state hamiltonian.
        i:
            index of initial hamiltonian, default 0.
        mode:
            the mode.
        '''
        dmi=self.dmlist[i]
        dmf=self.dmlist[f]
        Uitracker=dmi.Utracker
        Uftracker=dmf.Utracker
        bmtrackeri=dmi.bmtracker
        bmtrackerf=dmf.bmtracker
        Smat=csr_matrix([[1]])
        sl=[]
        N=self.N
        for k in xrange(N):
            t0=time.time()
            Uik=Uitracker.get(k)
            Ufk=Uftracker.get(k)
            bmi=bmtrackeri.get(k)
            bmf=bmtrackerf.get(k)
            if mode[0]=='t':
                Ufk=Ufk[:,bmf.kpmask]
            if mode[1]=='t':
                Uik=Uik[:,bmi.kpmask]
            Sexp=exp_blockize(Smat,block_marker=bmf,block_marker2=bmi)
            Smat=Ufk.T.conj().dot(Sexp.dot(Uik))
            sl.append(Smat)
            if mode[0]!='t':
                Smat=Smat[bmf.kpmask]
            if mode[1]!='t':
                Smat=Smat[:,bmi.kpmask]
            t1=time.time()
            print 'Updated Projection Matrix m = %s, Quality:%s(time cost:%s)'%(k,Smat.multiply(Smat).sum()/sqrt(Smat.shape[0]*Smat.shape[1]),t1-t0)
        self.S_list[(f,i)]=sl

    @property
    def N(self):
        '''expansion level'''
        return self.hlist[0].N

    @property
    def Nt(self):
        '''number of hamiltonians'''
        return len(self.hlist)

    def get_expect(self,opname,dt,f,i=0):
        '''
        get the expectation value of specific operator.

        opname:
            the name of operator.
        dt:
            the time.
        i/f:
            the index of initial/final hamiltonian.
        '''
        print 'Measuring operator %s h%d(t=0) -> h%d(t=%s)'%(opname,i,f,dt)
        hf=self.hlist[f]
        hi=self.hlist[i]
        otracker=hf.trackers[opname]
        kptracker=hf.trackers['kpmask']
        res=0
        N=self.N
        for m in xrange(N):
            #m is the chain lengtkpmaskh
            t0=time.time()
            O=csr_matrix(otracker.get(m))  #$%
            kpmask=kptracker.get(m)
            rho=self.rho_evolve(i,f,dt,m)
            truncmask=~kpmask
            if m==N-1:
                #for the last iteration, all the states are discarded
                truncmask,kpmask=kpmask,truncmask
            t1=time.time()
            OR=O.multiply(rho.T).tocsc()
            truncindices=where(truncmask)[0]
            kpindices=where(kpmask)[0]
            if len(truncindices)>0:
                #first part - trunc,trunc
                out1 = OR[:,truncindices]
                out2 = out1.tocsr()[truncindices,:]
                res+=out2.sum()
            if len(kpindices)>0 and len(truncindices)>0:
                #second part - trunc,keep
                out1 = OR[:,kpindices]
                out2 = out1.tocsr()[truncindices,:]
                res+=out2.sum()
                #third part - keep,trunc
                out1 = OR[:,truncindices]
                out2 = out1.tocsr()[kpindices,:]
                res+=out2.sum()
            t2=time.time()
            print 'Measured %s for iteration %s(time cost: e -> %s, s -> %s).'%(opname,m,t1-t0,t2-t1)
        return res

    def rho_evolve(self,i,f,dt,m):
        '''
        get the rho evolving from i to f with time dt.

        i/f:
            index of initial/final hamiltonian.
        dt:
            the evolution time.
        m:
            the iteration.
        '''
        rho0=self.get_rho0(i,m).tocsc()
        S=self.get_S(f,i,m).tocsr()
        t0=time.time()
        rho=S.dot(rho0).dot(S.T.conj())
        etracker=self.hlist[f].trackers['E_rescaled']
        E=etracker.get(m)
        DE=exp(-1j*dt*(E[...,newaxis]-E[newaxis,...]))
        rho=csr_matrix(rho.multiply(DE))
        t2=time.time()
        print 'evolving S%s,rho%s (cost:%s)'%(S.shape,rho0.shape,t2-t0)
        return rho

    def get_op(self,opname,i,m):
        '''
        get the operator of specific time and iteraction.

        opname:
            the name of operator.
        i:
            the hamiltonian index.
        m:
            the number of iteration.
        '''
        return self.hlist[i].trackers[opname].get(m)

    def get_S(self,f,i,m,mode='ft',**kwargs):
        '''
        get specific projection matrix.
        '''
        if not self.S_list.has_key((f,i)):
            print 'Initialing S-list for hamiltonian',i
            self.init_S_list(f,i,mode)
        S=self.S_list[f,i][m]
        return S

    def get_rho(self,i,m):
        '''
        get specific density matrix.

        i/m:
            specify the i-th time slice, m-th iteration.
        '''
        dm=self.dmlist[i]
        if not (dm.rholist is None):
            print 'Initialing rho-list for hamiltonian',i
            self.init_rho_list(i)
        rho=dm.rholist[m]
        return rho

    def get_oplist(self,opname,tindex):
        '''
        get a list of operator of specific time

        opname:
            the name of operator.
        tindex:
            the time index.
        '''
        return self.hlist[tindex].trackers[opname].data

    def rho_eq(self,T=0.,degeneracy_eps=1e-10):
        '''
        get equilibrium density matrix connecting f and i.

        T:
            temperature.
        degeneracy_eps:
            under which energy to view it as degenerate.
        '''
        return self.hlist[0].rho(T,diagonly=False,degeneracy_eps=degeneracy_eps)

class TDNRGManager(object):
    '''
    a manager class for TD-NRG.

    scale:
        the scale.
    tls:
        a list of timelines defined on multiple z scales.
    '''
    def __init__(self,scale,tls):
        self.tls=tls
        self.scale=scale
        self.setting={
                }

    def init_rho_list(self,i=0):
        '''
        initialze density matrix of hindex i.

        i:
            the hamiltonian index in timeline.
        '''
        for tl in self.tls:
            tl.init_rho_list(i)

    def init_S_list(self,f,i=0):
        '''
        initialze S-matrix projecting i to f

        f/i:
            target/source hamiltonian index.
        '''
        for tl in self.tls:
            tl.init_S_list(f,i)

def gettdmanager(chains,rdm=False):
    '''
    get a time line from a list of chain.

    chains:
        a list of chain.
    mops_a:
        a list of measurables.
    '''
    scale=chains[0].scale
    TL=[]
    tlist=array([chain.time for chain in chains])
    MNG=RDMManager if rdm else FDMManager
    for iz in xrange(scale.nz):
        dmlist=[MNG(chain.HN[iz]) for chain in chains]
        tl=RGTimeLine(dmlist,tlist)
        TL.append(tl)
    tdm=TDNRGManager(scale,TL)
    return tdm

def getdmmanagers(chain,T,rdm=False,degeneracy_eps=1e-15):
    '''
    get a DMManager instance from a list of chain.

    chain:
        a list of chain.
    T:
        the temperature.
    mops_a:
        a list of measurables.
    degeneracy_eps:
        the gate value to view energy level the same.
    '''
    MNG=RDMManager if rdm else FDMManager
    dml=[MNG(chain.HN[i],T=T,degeneracy_eps=degeneracy_eps) for i in xrange(len(chain.HN))]
    return dml

