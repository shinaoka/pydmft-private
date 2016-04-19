import numpy as np
import sys
import time
import os
from math import sin, cos, pi, sqrt
from lib import *
from scipy.fftpack import fftn, ifftn
import copy
import numpy.linalg as linalg

class DownFolding:
    parms={}
    ndim=0 #Spatial dimension
    kvec=None
    int_kvec=None
    kvec_list=None
    ek=np.zeros((0,0),dtype=float)
    wf=np.zeros((0,0,0),dtype=complex)
    epsilon=None
    W0q=None
    Wq=None
    Pmat=None
    omega=None
    tagert_band=0

    def __init__(self,parms):
        self.parms=copy.copy(parms)
        assert ('target_bands' in parms) ^ ('target_band' in parms)

        if 'target_band' in parms:
            self.target_bands = [parms['target_band']]
        elif 'target_bands' in parms:
            self.target_bands = copy.deepcopy(parms['target_bands'])

    def get_kindex(self,int_kvec):
        raise RuntimeError("get_kindex must be implemented in a subclass.")
    
    def get_Hk(self,kvec):
        raise RuntimeError("get_Hk must be implemented in a subclass.")

    def plus_k(self,ik1,ik2):
        ndiv = self.parms['ndiv_k']
        kvec_t = self.int_kvec[ik1]+self.int_kvec[ik2]
        for i in range(self.ndim):
            kvec_t[i] = (kvec_t[i]+100*ndiv)%ndiv
        return self.get_kindex(kvec_t)

    #Silly and slow implimentation
    def inv_k(self,ik):
        ndiv = self.parms['ndiv_k']
        kvec_t = -self.int_kvec[ik]
        for i in range(self.ndim):
            kvec_t[i] = (kvec_t[i]+100*ndiv)%ndiv
        return self.get_kindex(kvec_t)

    #Silly and slow implimentation
    def ft_to_k(self,fr):
        nk=self.kvec.shape[0]
        if self.ndim == len(fr.shape):
            return nk*ifftn(fr)
        else:
            shape_org = fr.shape
            ndiv = self.parms['ndiv_k']
            if self.ndim==2:
                view = fr.reshape((ndiv,ndiv))
                return nk*ifftn(view).reshape(shape_org)
            elif  self.ndim==3:
                view = fr.reshape((ndiv,ndiv,ndiv))
                return nk*ifftn(view).reshape(shape_org)
            else:
                raise RuntimeError("invalid fr")

    def ft_to_r(self,fk):
        nk=self.kvec.shape[0]
        if self.ndim == len(fk.shape):
            return fftn(fk)/nk
        else:
            shape_org = fk.shape
            ndiv = self.parms['ndiv_k']
            if self.ndim==2:
                view = fk.reshape((ndiv,ndiv))
                return fftn(view).reshape(shape_org)/nk
            elif  self.ndim==3:
                view = fk.reshape((ndiv,ndiv,ndiv))
                return fftn(view).reshape(shape_org)/nk
            else:
                raise RuntimeError("invalid fk")

    def Init_band(self):
        raise RuntimeError("__Init_band of Downfolding must be overrode.")

    def Init_Interaction(self):
        raise RuntimeError("This function must be overrode.")

    def Cal_screened_interaction_slow(self):
        parms=self.parms
        nomega=parms['nomega']
        nk=parms['nk']
        norb=parms['norb']

        self.Pmat=np.zeros((nomega,nk,norb,norb),dtype=complex)
        self.Wr=np.zeros((nomega,nk,norb,norb),dtype=complex)
        self.Wq=np.zeros((nomega,nk,norb,norb),dtype=complex)
        self.omega=np.zeros((nomega,),dtype=float)

        if nomega==1:
            self.omega[0]=parms['emin']
        else:
            for iomega in range(nomega):
                self.omega[iomega]=(iomega*(parms['emax']-parms['emin']))/(nomega-1)+parms['emin']

        zdelta=1J*parms['delta']
        ef=parms['ef']
        dbeta=1.0/parms['delta']

        for iq in range(nk):
            for ik in range(nk):
                ikq=self.plus_k(ik,iq)
                for n in range(norb): #n: occupied bands
                    if self.ek[ikq,n]>ef:
                        continue
                    for m in range(norb): #m: unoccupied bands
                        if self.ek[ik,m]<=ef:
                            continue

                        if n in self.target_bands and m in self.target_bands: #constrained RPA (removing band0-band0 polarization)
                            continue

                        za1=1.0/(self.omega[:]-self.ek[ik,m]+self.ek[ikq,n]+zdelta)
                        za2=1.0/(self.omega[:]+self.ek[ik,m]-self.ek[ikq,n]-zdelta)
                        for alpha in range(norb):
                            for beta in range(norb):
                                ztmp = \
                                    self.wf[ik,n,alpha].conjugate()* \
                                    self.wf[ikq,m,alpha]* \
                                    self.wf[ik,n,beta]* \
                                    self.wf[ikq,m,beta].conjugate()

                                self.Pmat[:,iq,alpha,beta] += ztmp*za1-ztmp.conjugate()*za2
        self.Pmat=self.Pmat/nk

        for iq in range(nk):
            for iomega in range(nomega):
                if self.nspin==1:
                    self.epsilon[iomega,iq,:,:]=np.linalg.inv(np.identity(norb)-2*np.dot(self.W0q[iq],self.Pmat[iomega,iq]))
                else:
                    self.epsilon[iomega,iq,:,:]=np.linalg.inv(np.identity(norb)-1*np.dot(self.W0q[iq],self.Pmat[iomega,iq]))
                self.Wq[iomega,iq,:,:]=np.dot(self.epsilon[iomega,iq],self.W0q[iq])
                for iorb in range(norb):
                    for jorb in range(norb):
                        self.Wr[iomega,0:nk,iorb,jorb]=self.ft_to_r(self.Wq[iomega,0:nk,iorb,jorb])

    def Cal_screened_interaction(self):
        parms=self.parms
        nomega=parms['nomega']
        nk=parms['nk']
        norb=parms['norb']

        self.Pmat=np.zeros((nomega,nk,norb,norb),dtype=complex)
        self.Wr=np.zeros((nomega,nk,norb,norb),dtype=complex)
        self.Wq=np.zeros((nomega,nk,norb,norb),dtype=complex)
        self.omega=np.zeros((nomega,),dtype=float)

        if nomega==1:
            self.omega[0]=parms['emin']
        else:
            for iomega in range(nomega):
                self.omega[iomega]=(iomega*(parms['emax']-parms['emin']))/(nomega-1)+parms['emin']

        zdelta=1J*parms['delta']
        ef=parms['ef']
        dbeta=1.0/parms['delta']

        bmat = np.zeros([norb,nk*norb**2],dtype=complex)
        D = np.zeros([norb**2,nk*norb**2],dtype=complex)
        C1 = np.zeros([nk*norb**2,nomega],dtype=complex)
        C2 = np.zeros([nk*norb**2,nomega],dtype=complex)

        for iq in range(nk):
            nalpha = 0
            time1 = time.clock()
            for ik in range(nk):
                ikq=self.plus_k(ik,iq)
                for n in range(norb): #n: occupied bands
                    if self.ek[ik,n]>ef:
                        continue
                    for m in range(norb): #m: unoccupied bands
                        if self.ek[ikq,m]<=ef:
                            continue

                        if n in self.target_bands and m in self.target_bands: #constrained RPA (removing polarizations within target manifold)
                            continue

                        #if iq==0:
                            #print "debug ik ", ik, " n, m ", n, m

                        bmat[0:norb,nalpha] = self.wf[ik,n,0:norb].conjugate()*self.wf[ikq,m,0:norb]
                        dE = self.ek[ikq,m]-self.ek[ik,n]
                        if dE<0:
                            raise RuntimeError("Excitation energy is negative!")
                        C1[nalpha,:]=1.0/(self.omega[:]-dE+zdelta)
                        C2[nalpha,:]=1.0/(self.omega[:]+dE-zdelta)
                        nalpha += 1

            time2 = time.clock()
            j=0
            for iorb1 in range(norb):
                for iorb2 in range(norb):
                    D[j,0:nalpha] = np.multiply(bmat[iorb1,0:nalpha],bmat[iorb2,0:nalpha].conjugate())
                    j += 1

            time3 = time.clock()
            Pmat_t = np.dot(D,C1)
            Pmat_t -= np.dot(D.conjugate(),C2)
            time4 = time.clock()
            self.Pmat[:,iq,:,:] += Pmat_t.reshape((norb,norb,nomega)).transpose((2,0,1))
            time5 = time.clock()


        self.Pmat=self.Pmat/nk

        for iq in range(nk):
            for iomega in range(nomega):
                if self.nspin==1:
                    self.epsilon[iomega,iq,:,:]=np.linalg.inv(np.identity(norb)-2*np.dot(self.W0q[iq],self.Pmat[iomega,iq]))
                else:
                    self.epsilon[iomega,iq,:,:]=np.linalg.inv(np.identity(norb)-1*np.dot(self.W0q[iq],self.Pmat[iomega,iq]))
                self.Wq[iomega,iq,:,:]=np.dot(self.epsilon[iomega,iq],self.W0q[iq])
                for iorb in range(norb):
                    for jorb in range(norb):
                        self.Wr[iomega,0:nk,iorb,jorb]=self.ft_to_r(self.Wq[iomega,0:nk,iorb,jorb])

    def Wannier_fit(self):
        nk=self.parms['nk']
        norb=self.parms['norb']
        self.Uwfk,self.Uwfr,self.Trans=oneshot_Uwfk2(norb,nk,self.wf,self.ek)
        self.Transk=np.zeros_like(self.Trans,dtype=complex)
        for iwann in range(norb):
            self.Transk[iwann,:]=self.ft_to_k(self.Trans[iwann,:])

    def get_wf_density(self,iwann):
        nk=self.parms['nk']
        norb=self.parms['norb']

        density=np.zeros((nk,norb),dtype=float)
        for ir in range(nk):
            for iorb in range(norb):
                density[ir,iorb]=abs(self.Uwfr[iwann,ir,iorb])**2
        return density

    def Project_Interaction_each(self,Wr_t):
        nk=self.parms['nk']
        norb=self.parms['norb']

        Vr_t=np.zeros((nk,norb,norb),dtype=complex)
        Vq_t=np.zeros((nk,norb,norb),dtype=complex)
        for iR in range(nk):
            for iwann1 in range(norb):
                n1=self.get_wf_density(iwann1,iR)

                for iwann2 in range(norb):
                    n2=self.get_wf_density(iwann2,0)

                    for ir1 in range(nk):
                        for ir2 in range(nk):
                            #ir=(ir1-ir2+2*nk)%nk
                            ir=self.plus_k(ir1,self.inv_k(ir2))
                            for iorb1 in range(norb):
                                for iorb2 in range(norb):
                                    Vr_t[iR,iwann1,iwann2]+=Wr_t[ir,iorb1,iorb2]*n1[ir1,iorb1]*n2[ir2,iorb2]

        for iorb1 in range(norb):
            for iorb2 in range(norb):
                Vq_t[0:nk,iorb1,iorb2]=self.ft_to_k(Vr_t[0:nk,iorb1,iorb2])

        return Vr_t,Vq_t

    def Project_Interaction(self):
        nk=self.parms['nk']
        norb=self.parms['norb']
        nomega=self.parms['nomega']

        self.Vr=np.zeros((nomega,nk,norb,norb),dtype=complex)
        self.Vq=np.zeros((nomega,nk,norb,norb),dtype=complex)
        for iomega in range(nomega):
            #rt,qt = self.Project_Interaction_each(self.Wr[iomega,:,:,:])
            rt,qt = self.Project_Interaction_each2(self.Wq[iomega,:,:,:])
            self.Vr[iomega,:,:,:] = 1.0*rt
            self.Vq[iomega,:,:,:] = 1.0*qt

        #self.V0r,self.V0q = self.Project_Interaction_each(self.W0r)
        self.V0r,self.V0q = self.Project_Interaction_each2(self.W0q)

    def Project_Interaction_each2(self,Wq_t):
        nk=self.parms['nk']
        norb=self.parms['norb']

        V0r=np.zeros((nk,norb,norb),dtype=complex)
        V0q=np.zeros((nk,norb,norb),dtype=complex)
        density_wann_k=np.zeros((norb,norb,nk),dtype=complex)
        inv_ik=np.zeros((nk),dtype=int)

        for ik in range(nk):
            inv_ik[ik] = self.inv_k(ik)

        #calculate density in k space
        for iwann1 in range(norb):
            t_dens=self.get_wf_density(iwann1)
            for iorb1 in range(norb):
                density_wann_k[iwann1,iorb1,:]=self.ft_to_k(t_dens[:,iorb1])


        for iwann1 in range(norb):
            for iwann2 in range(norb):

                for iorb1 in range(norb):
                    for iorb2 in range(norb):
                        for ik in range(nk):
                            V0q[ik,iwann1,iwann2] += \
                                    density_wann_k[iwann1,iorb1,ik]*\
                                    Wq_t[inv_ik[ik],iorb1,iorb2]*\
                                    density_wann_k[iwann2,iorb2,inv_ik[ik]]

        for iwann1 in range(norb):
            for iwann2 in range(norb):
                V0r[0:nk,iwann1,iwann2]=self.ft_to_r(V0q[0:nk,iwann1,iwann2])

        return V0r,V0q

class DownFolding1D(DownFolding):
    def __init__(self,parms):
        self.ndim=1
        self.nspin = 1
        if "nspin" in parms and parms["nspin"]==2:
            self.nspin = 2
        DownFolding.__init__(self,parms)

    def get_kindex(self,int_kvec):
        return int_kvec[0]

    def get_Hk(self,kvec):
        nk=self.parms['nk']
        assert nk%2==0
        norb=self.parms['norb']
        hk=np.zeros((norb,norb),dtype=complex)
        hk[:,:] = 1.0*self.Hn0[0,:,:]
        for ir in range(1,nk):
            hk[:,:] += self.Hn0[ir,:,:]*np.exp(-1J*kvec[0]*ir) 
        return hk

    def Init_band(self):
        parms=self.parms
        nk=parms['nk']
        norb=parms['norb']
        self.kvec=np.zeros((nk,self.ndim),dtype=float)
        self.int_kvec=np.zeros((nk,self.ndim),dtype=int)
        self.ek=np.zeros((nk,norb),dtype=float)
        self.wf=np.zeros((nk,norb,norb),dtype=complex)
        self.Hn0=np.zeros((nk,norb,norb),dtype=float)

        self.kvec_list = []
        for ik in range(nk):
            self.kvec[ik,0]=(2*pi/nk)*ik
            self.int_kvec[ik,0]=ik
            self.kvec_list.append([ik])

        for iorb in range(norb):
            self.Hn0[0,iorb,iorb]=parms['e'+str(iorb)]
            self.Hn0[1,iorb,iorb]=-parms['t'+str(iorb)]
            self.Hn0[nk-1,iorb,iorb]=-parms['t'+str(iorb)]

        if "nspin" in parms and parms["nspin"]==2:
            assert "eh_symm" in parms and parms["eh_symm"] != 0
            for iorb1 in range(norb):
                for iorb2 in range(norb):
                    if iorb1/2 != iorb2/2 and (iorb1 in self.target_bands or iorb2 in self.target_bands):
                        if (iorb1%2)==(iorb2%2):
                            self.Hn0[0,iorb1,iorb2]=-parms['tp']
        else:
            if "eh_symm" in parms and parms["eh_symm"] != 0:
                for iorb1 in range(norb):
                    for iorb2 in range(norb):
                        if iorb1 != iorb2 and (iorb1 in self.target_bands or iorb2 in self.target_bands):
                            self.Hn0[0,iorb1,iorb2]=-parms['tp']
            else:
                for iorb1 in range(norb):
                    for iorb2 in range(norb):
                        if iorb1 != iorb2:
                            self.Hn0[0,iorb1,iorb2]=-parms['tp']

        for ik in range(nk):
            hk = self.get_Hk(self.kvec[ik,:])
            if self.nspin==1:
                evals,evec = eigh_ordered(hk)
            else:
                evals,evec = eigh_ordered_spin(hk)
            for ie in range(norb):
                self.ek[ik,ie]=copy.deepcopy(evals[ie])
                for i in range(norb):
                    self.wf[ik,ie,i]=evec[i,ie]

    def Init_Interaction(self):
        nk=self.parms['nk']
        norb=self.parms['norb']
        nomega=self.parms['nomega']

        self.W0q=np.zeros((nk,norb,norb),dtype=complex)
        self.W0r=np.zeros((nk,norb,norb),dtype=complex)
        self.epsilon=np.zeros((nomega,nk,norb,norb),dtype=complex)

        for ik in range(nk):
            for iorb1 in range(norb):
                for iorb2 in range(norb):
                    tkey='U_'+str(max(iorb1,iorb2))+'_'+str(min(iorb1,iorb2))
                    if not self.parms.has_key(tkey):
                        continue
                    self.W0q[ik,iorb1,iorb2]+=self.parms[tkey]

            for iorb1 in range(norb):
                for iorb2 in range(norb):
                    tkey='V_'+str(max(iorb1,iorb2))+'_'+str(min(iorb1,iorb2))+'_nn'
                    if not self.parms.has_key(tkey):
                        continue
                    self.W0q[ik,iorb1,iorb2]+=self.parms[tkey]*cos(self.kvec[ik,0])*2

        for iorb1 in range(norb):
            for iorb2 in range(norb):
                self.W0r[0:nk,iorb1,iorb2]=self.ft_to_r(self.W0q[0:nk,iorb1,iorb2]).real

class DownFolding2D(DownFolding):
    def __init__(self,parms):
        DownFolding.__init__(self,parms)
        self.ndim=parms['ndim']
        self.nspin = 1
        if "nspin" in parms and parms["nspin"]==2:
            self.nspin = 2

    def get_kindex(self,int_kvec):
        if self.ndim==1:
            return int_kvec[0]
        elif self.ndim==2:
            return self.parms['ndiv_k']*int_kvec[0] + int_kvec[1]
        elif self.ndim==3:
            ndiv_k = self.parms['ndiv_k']
            return (ndiv_k**2)*int_kvec[0] + ndiv_k*int_kvec[1] + int_kvec[2]
        else:
            raise RuntimeError("Unsupported dimension")

    def get_Hk(self,kvec):
        nk=self.parms['nk']
        assert nk%2==0
        norb=self.parms['norb']
        hk=np.zeros((norb,norb),dtype=complex)
        for ir in range(nk):
            hk[:,:] += self.Hn0[ir,:,:]*np.exp(-1J*np.dot(kvec,self.int_kvec[ir]) )
        return hk

    def __dist(self,ivec1,ivec2):
        ndiv= self.parms['ndiv_k']
        ivec_r = np.zeros_like(ivec1)
        for i in range(self.ndim):
            ivec_r[i] = abs(ivec1[i]-ivec2[i])%ndiv
            ivec_r[i] = min(ivec_r[i], ndiv-ivec_r[i])
        return ivec_r

    def Init_band(self):
        parms=self.parms
        if parms['ndiv_k']**self.ndim != parms['nk']:
            raise RuntimeError("ndiv_k and nk are inconsistent.")
        self.nk=parms['nk']
        nk=self.nk
        norb=parms['norb']
        self.ek=np.zeros((nk,norb),dtype=float)
        self.wf=np.zeros((nk,norb,norb),dtype=complex)

        #generate a list of kvec
        ndiv= parms['ndiv_k']
        self.inv_ik=np.zeros((nk,),dtype=int)
        self.kvec_list = []

        self.int_kvec = mk_map(ndiv,self.ndim)
        self.kvec = (2*pi/ndiv)*self.int_kvec
        for ik in range(nk):
            self.kvec_list.append(self.int_kvec[ik,:].tolist())

        for ik in range(nk):
            tmp = np.array(self.int_kvec[ik,:])
            tmp[:] = (-tmp[:]+10*ndiv)%ndiv
            self.inv_ik[ik] = self.kvec_list.index(tmp.tolist())

        z = self.parms['z']
        self.Hn0=np.zeros((nk,norb,norb),dtype=float)
        for iorb in range(norb):
            self.Hn0[0,iorb,iorb]=parms['e'+str(iorb)]

        if "nspin" in parms and parms["nspin"]==2:
            assert "eh_symm" in parms and parms["eh_symm"] != 0
            for iorb1 in range(norb):
                for iorb2 in range(norb):
                    if iorb1/2 != iorb2/2 and (iorb1 in self.target_bands or iorb2 in self.target_bands):
                        if (iorb1%2)==(iorb2%2):
                            self.Hn0[0,iorb1,iorb2]=-parms['tp']
        else:
            if "eh_symm" in parms and parms["eh_symm"] != 0:
                for iorb1 in range(norb):
                    for iorb2 in range(norb):
                        if iorb1 != iorb2 and (iorb1 in self.target_bands or iorb2 in self.target_bands):
                            self.Hn0[0,iorb1,iorb2]=-parms['tp']
            else:
                for iorb1 in range(norb):
                    for iorb2 in range(norb):
                        if iorb1 != iorb2:
                            self.Hn0[0,iorb1,iorb2]=-parms['tp']

        for ir in range(1,nk):
            #NN
            dist_vec = self.__dist(self.int_kvec[ir,:], self.int_kvec[0,:])
            if np.sum(dist_vec)==1:
                self.Hn0[ir,:,:] = -np.identity(norb)
                for iorb in xrange(norb):
                    self.Hn0[ir,iorb,iorb] *= parms['t'+str(iorb)]

        for ik in range(nk):
            hk = self.get_Hk(self.kvec[ik,:])
            if self.nspin==1:
                evals,evec = eigh_ordered(hk)
            elif self.nspin==2:
                evals,evec = eigh_ordered_spin(hk)
            for ie in range(norb):
                self.ek[ik,ie]=copy.deepcopy(evals[ie])
                for i in range(norb):
                    self.wf[ik,ie,i]=evec[i,ie]

    def Init_Interaction(self):
        nk=self.parms['nk']
        norb=self.parms['norb']
        nomega=self.parms['nomega']

        self.W0q=np.zeros((nk,norb,norb),dtype=complex)
        self.W0r=np.zeros((nk,norb,norb),dtype=complex)
        self.epsilon=np.zeros((nomega,nk,norb,norb),dtype=complex)

        z = self.parms['z']
        for ik in range(nk):
            for iorb1 in range(norb):
                for iorb2 in range(norb):
                    tkey='U_'+str(max(iorb1,iorb2))+'_'+str(min(iorb1,iorb2))
                    if not self.parms.has_key(tkey):
                        continue
                    self.W0q[ik,iorb1,iorb2]+=self.parms[tkey]

            for iorb1 in range(norb):
                for iorb2 in range(norb):
                    tkey='V_'+str(max(iorb1,iorb2))+'_'+str(min(iorb1,iorb2))+'_nn'
                    if not self.parms.has_key(tkey):
                        #print "No key: ", tkey
                        continue
                    self.W0q[ik,iorb1,iorb2] += self.parms[tkey]*(\
                            cos(self.kvec[ik,0])+z*cos(self.kvec[ik,1])\
                        )*2.0

        for iorb1 in range(norb):
            for iorb2 in range(norb):
                self.W0r[0:nk,iorb1,iorb2]=self.ft_to_r(self.W0q[0:nk,iorb1,iorb2]).real
