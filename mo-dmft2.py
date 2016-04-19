import pyalps 
#import pyalps.cthyb as cthyb # solver module
#import pyalps.mpi as mpi     # MPI library
import numpy as np
from numpy import cos,pi
from numpy.linalg import inv
import sys
import time
import integration as intgr
import re 
import input_parms as inp
import os
import subprocess
import fourier_transform as ft
from lib import *
import scipy.optimize as sciopt
from hyb import call_hyb_matrix
from h5dump import *
import copy

def print_ek_info(Nk,norb,Hk):
    ek = np.zeros((Nk,norb),dtype=float)
    for ik in range(Nk):
        evals,evecs = eigh_ordered(Hk[ik,:,:])
        ek[ik,:] = evals
    print "ek=", np.mean(ek,0)

def read_text_data(fname,ndata,ndim,dtype_data,offset):
    f=open(fname,"r")
    data=np.zeros((ndata,ndim),dtype=dtype_data)
    for i in range(ndata):
        l = f.readline()
        line_elements = re.split('[\s\t)(,]+', l.strip())
        if dtype_data==float:
            for j in range(ndim):
                data[i][j]=float(line_elements[j+offset])
        if dtype_data==complex:
            for j in range(ndim):
                data[i][j]=float(line_elements[2*j+offset])+1J*float(line_elements[2*j+1+offset])
    return data

def ft_to_tau_hyb(ndiv_tau, beta, matsubara_freq, tau, Vhk, Vhk2, data_n, data_tau,cutoff):
 norb = Vhk.shape[0]
 #tail correction
 #Htmp = - np.dot(H1,np.dot(H1,H1))+2*H1*H2-H3
 #print "cutoff in ft_to_tau_hyb = ", cutoff
 for it in range(ndiv_tau+1):
     tau_tmp=tau[it]
     if it==0:
         tau_tmp=1E-4*(beta/ndiv_tau)
     if it==ndiv_tau:
         tau_tmp=beta-1E-4*(beta/ndiv_tau)
     ztmp=np.zeros((norb,norb),dtype=complex)
     for im in range(cutoff):
         ztmp+=(data_n[im]+1J*Vhk/matsubara_freq[im]-Vhk2/matsubara_freq[im]**2)*np.exp(-1J*matsubara_freq[im]*tau_tmp)
         #ztmp+=(data_n[im]+1J*Vhk/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
     ztmp=ztmp/beta
     data_tau[it]=2.0*ztmp.real-0.5*Vhk-0.25*Vhk2*(-beta+2*tau_tmp)

def calc_Glatt(beta,matsubara_freq,Hk,self_ene,vmu,cutoff):
    ndiv_tau = len(matsubara_freq)
    norb = self_ene.shape[1]
    G_latt=np.zeros((ndiv_tau,norb,norb),dtype=complex)
    for im in range(ndiv_tau):
        G_latt[im,:,:] = 0.0+0.0J
        digmat = (1J*matsubara_freq[im]+vmu)*np.identity(norb)
        for ik in range(Nk):
            G_latt[im,:,:] += inv(digmat-Hk[ik,:,:]-self_ene[im,:,:])
        G_latt[im,:,:] /= Nk
        #print im, G_latt[im,0,0].imag

    tau_tmp = beta-1E-4*(beta/ndiv_tau)
    ztmp = 0.0
    for im in range(cutoff):
        for iorb in range(norb):
            ztmp += (G_latt[im,iorb,iorb]+1J/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
    rtmp = 2.0*ztmp.real/beta-0.5*norb
    
    return G_latt,-rtmp

def calc_Glatt2(beta,matsubara_freq,Hk_rlist,Hk_wlist,self_ene,vmu,cutoff):
    ndiv_tau = len(matsubara_freq)
    norb = self_ene.shape[1]
    G_latt=np.zeros((ndiv_tau,norb,norb),dtype=complex)
    Nk_reduced = Hk_rlist.shape[0]
    for im in range(ndiv_tau):
        G_latt[im,:,:] = 0.0+0.0J
        digmat = (1J*matsubara_freq[im]+vmu)*np.identity(norb)
        ksum = 0
        for ik in range(Nk_reduced):
            G_latt[im,:,:] += inv(digmat-Hk_rlist[ik,:,:]-self_ene[im,:,:])*Hk_wlist[ik]
            ksum += Hk_wlist[ik]
        G_latt[im,:,:] /= Nk
        if (ksum != Nk):
            raise RuntimeError("Fatal error in calc_Glatt")

    tau_tmp = beta-1E-4*(beta/ndiv_tau)
    ztmp = 0.0
    for im in range(cutoff):
        for iorb in range(norb):
            ztmp += (G_latt[im,iorb,iorb]+1J/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
    rtmp = 2.0*ztmp.real/beta-0.5*norb
    
    return G_latt,-rtmp

def calc_Glatt2(beta,matsubara_freq,Hk_rlist,Hk_wlist,self_ene,vmu,cutoff):
    ndiv_tau = len(matsubara_freq)
    norb = self_ene.shape[1]
    G_latt=np.zeros((ndiv_tau,norb,norb),dtype=complex)
    Nk_reduced = Hk_rlist.shape[0]
    for im in range(ndiv_tau):
        G_latt[im,:,:] = 0.0+0.0J
        digmat = (1J*matsubara_freq[im]+vmu)*np.identity(norb)
        ksum = 0
        for ik in range(Nk_reduced):
            G_latt[im,:,:] += inv(digmat-Hk_rlist[ik,:,:]-self_ene[im,:,:])*Hk_wlist[ik]
            ksum += Hk_wlist[ik]
        G_latt[im,:,:] /= Nk
        if (ksum != Nk):
            raise RuntimeError("Fatal error in calc_Glatt")

    tau_tmp = beta-1E-4*(beta/ndiv_tau)
    ztmp = 0.0
    for im in range(cutoff):
        for iorb in range(norb):
            ztmp += (G_latt[im,iorb,iorb]+1J/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
    rtmp = 2.0*ztmp.real/beta-0.5*norb
    
    return G_latt,-rtmp

def calc_Glatt3(beta,matsubara_freq,Hk_rlist,Hk_wlist,self_ene,vmu,cutoff,Hk_mean):
    ndiv_tau = len(matsubara_freq)
    norb = self_ene.shape[1]
    G_latt=np.zeros((ndiv_tau,norb,norb),dtype=complex)
    Nk_reduced = Hk_rlist.shape[0]
    for im in range(ndiv_tau):
        G_latt[im,:,:] = 0.0+0.0J
        digmat = (1J*matsubara_freq[im]+vmu)*np.identity(norb)
        ksum = 0
        for ik in range(Nk_reduced):
            G_latt[im,:,:] += inv(digmat-Hk_rlist[ik,:,:]-self_ene[im,:,:])*Hk_wlist[ik]
            ksum += Hk_wlist[ik]
        G_latt[im,:,:] /= Nk
        if (ksum != Nk):
            raise RuntimeError("Fatal error in calc_Glatt")

    dtau = 1E-4*(beta/ndiv_tau)
    tau_list = [dtau, beta-dtau]
    G_tau = []
    for tau in tau_list:
        ztmp = 0.0
        for im in range(cutoff):
            for iorb in range(norb):
                ztmp += (G_latt[im,iorb,iorb]+1J/matsubara_freq[im]+Hk_mean[iorb,iorb]/matsubara_freq[im]**2)*np.exp(-1J*matsubara_freq[im]*tau)
        rtmp = 2.0*ztmp.real/beta
        for iorb in range(norb):
            rtmp += -0.5 + 0.25*Hk_mean[iorb,iorb]*(-beta+2*tau)
        G_tau.append(rtmp)
    print G_tau
    ntot = ((G_tau[0]-G_tau[1]))/2+norb/2.0
    
    return G_latt, ntot

#### setup parameters ####
app_parms = inp.read_input_parms(sys.argv[1]+'.h5')
app_parms['prefix'] = sys.argv[1]

norb=app_parms["N_ORB"]
nkdiv=app_parms["N_K_DIV"]
Nk=nkdiv**3 #3D DOS
PM=True
vmix=1.0
vmu=app_parms["MU"]
tote=app_parms["N_ELEC"]
vbeta=app_parms["BETA"]
vconverged=app_parms["CONVERGED"]
ndiv_tau=app_parms["NMATSUBARA"]
cutoff_fourie=app_parms["CUTOFF_FOURIE"]

matsubara_freq=np.zeros((ndiv_tau,),dtype=float)
matsubara_freq_boson=np.zeros((ndiv_tau,),dtype=float)
tau=np.zeros((ndiv_tau+1,),dtype=float)

# uncorrelated lattice Green function
for im in range(ndiv_tau):
    matsubara_freq[im]=((2*im+1)*np.pi)/vbeta
for it in range(ndiv_tau+1):
    tau[it]=(vbeta/ndiv_tau)*it

#Initialization of H(k)
H0 = np.zeros((norb,norb),dtype=float)#We assume these are real.
if 'EH_HYB' in app_parms and app_parms['EH_HYB'] != 0:
    target = app_parms['TARGET_ORBITAL']
    for iorb in range(norb):
        for iorb2 in range(norb):
            if iorb!=iorb2 and (iorb==target or iorb2==target):
                H0[iorb,iorb2] = -app_parms["t'"]
else:
    for iorb in range(norb):
        for iorb2 in range(norb):
            if iorb!=iorb2:
                H0[iorb,iorb2] = -app_parms["t'"]


for iorb in range(norb):
    H0[iorb,iorb] = app_parms["E"+str(iorb)]

Hnn = np.zeros((norb,norb),dtype=float)
for i in range(norb):
    Hnn[i,i] = -1.0

#H0
evals,evecs = eigh_ordered(H0)
print "H0 = ", H0
print "Evals of H0 = "
print evals

Hk = np.zeros((Nk,norb,norb),dtype=float)
Hk2 = np.zeros((Nk,norb,norb),dtype=float) #H(k)*H(k): not element-wise
Hk3 = np.zeros((Nk,norb,norb),dtype=float) #H(k)*H(k)*H(k): not element-wise
ik = 0
cos_list = {}
for i in range(nkdiv):
    for j in range(nkdiv):
        for k in range(nkdiv):
            Hk[ik,:,:] = H0[:,:]+2.0*app_parms["t"]*(cos(2*i*pi/nkdiv)+cos(2*j*pi/nkdiv)+cos(2*k*pi/nkdiv))*Hnn[:,:]
            Hk2[ik,:,:] = np.dot(Hk[ik,:,:],Hk[ik,:,:])
            Hk3[ik,:,:] = np.dot(Hk[ik,:,:],np.dot(Hk[ik,:,:],Hk[ik,:,:]))
            cos_sum = cos(2*i*pi/nkdiv)+cos(2*j*pi/nkdiv)+cos(2*k*pi/nkdiv)
            if cos_sum in cos_list:
                cos_list[cos_sum] += 1
            else:
                cos_list[cos_sum] = 1
            ik+=1
print "Num of equivalent k points =", len(cos_list)
Hk_reduced_list = np.zeros((len(cos_list),norb,norb),dtype=float)
Hk_w_reduced_list = np.zeros((len(cos_list),),dtype=int)
ik=0
for k,v in cos_list.items():
    Hk_reduced_list[ik,:,:] = H0[:,:]+2.0*app_parms["t"]*Hnn[:,:]*k
    Hk_w_reduced_list[ik]=v
    ik+=1

Hk_mean = np.zeros((norb,norb),dtype=float)
Hk_var = np.zeros((norb,norb),dtype=float)
Hk_2nd = np.zeros((norb,norb),dtype=float)
Hk_3rd = np.zeros((norb,norb),dtype=float)
for iorb in range(norb):
    for jorb in range(norb):
        Hk_mean[iorb,jorb] = np.mean(Hk[:,iorb,jorb])
        Hk_var[iorb,jorb] = np.mean(Hk2[:,iorb,jorb])
        Hk_2nd[iorb,jorb] = np.mean(Hk2[:,iorb,jorb])
        Hk_3rd[iorb,jorb] = np.mean(Hk3[:,iorb,jorb])
Hk_var[:,:] -= np.dot(Hk_mean,Hk_mean)
print "Hk_mean=", Hk_mean
print "Hk_var=", Hk_var
Hk_var2 = np.zeros_like(Hk_var)
#Hk_var2 = np.zeros_like(Hk_var)
print "Hk_var2=", Hk_var2

print_ek_info(Nk,norb,Hk)

#rotation matrix of hybridization function
evals,evecs=eigh_ordered(Hk_mean)
if 'BASIS_ROT_MAT' in app_parms:
    rotmat_hyb = 1.*app_parms['BASIS_ROT_MAT']
    print 'Using BASIS_ROT_MAT ', rotmat_hyb
    assert rotmat_hyb.shape[0] == norb
    assert rotmat_hyb.shape[1] == norb
elif 'BASIS_ROT' in app_parms and app_parms['BASIS_ROT']!=0:
    rotmat_hyb = 1.0*evecs
else:
    rotmat_hyb = np.identity(norb,dtype=float)

if 'mu' in app_parms:
    raise RuntimeError("Do not use mu")

isc = 0

#### self-consistent loop ####
def calc_diff(self_ene_dmp_in):
    global isc, vmu

    self_ene_in = np.array(self_ene_dmp_in)
    for iorb in range(norb):
        for iorb2 in range(norb):
            self_ene_in[:,iorb,iorb2] = self_ene_dmp_in[:,iorb,iorb2]/dmp_fact

    #### Lattice Green function ####
    nite = 5
    if 'N_CHEM_LOOP' in app_parms:
        nite = app_parms['N_CHEM_LOOP']
    if app_parms['OPT_MU']<=0.0:
        nite = 0
    if 'FIX_MU' in app_parms:
        if isc <= app_parms['FIX_MU']:
            nite = 0
    for i_chem in range(nite): #By default, we do it 5 times.
        time1 = time.clock()
        G_latt,tote_tmp = calc_Glatt3(vbeta,matsubara_freq,Hk_reduced_list,Hk_w_reduced_list,self_ene_in,vmu,cutoff_fourie,Hk_mean)
        time2 = time.clock()
        print "Computing G_latt(tau) took ", time2-time1, " sec."
        print " mu = ", vmu, tote_tmp
        sys.stdout.flush()
        vmu = (vmu-np.abs(app_parms['OPT_MU'])*(tote_tmp-app_parms["N_ELEC"])).real
    print "Computing lattice Green's function..."
    G_latt,tote_tmp = calc_Glatt3(vbeta,matsubara_freq,Hk_reduced_list,Hk_w_reduced_list,self_ene_in,vmu,cutoff_fourie,Hk_mean)
    print " mu = ", vmu, tote_tmp
    sys.stdout.flush()
    #print "debug tote_tmp ", tote_tmp

    #np.savetxt(app_parms["prefix"]+"-G_latt00-isc"+str(isc)+".txt",G_latt[:,0,0].imag)
    #np.savetxt(app_parms["prefix"]+"-G_latt01-isc"+str(isc)+".txt",G_latt[:,0,1].imag)
    #np.savetxt(app_parms["prefix"]+"-G_latt11-isc"+str(isc)+".txt",G_latt[:,1,1].imag)
    #np.savetxt(app_parms["prefix"]+"-G_latt22-isc"+str(isc)+".txt",G_latt[:,2,2].imag)

    #### Cavity Green's function ####
    print "Computing cavity Green's function..."
    invG0 = np.zeros((ndiv_tau,norb,norb),dtype=complex)
    for im in range(ndiv_tau):
        invG0[im,:,:]=inv(G_latt[im,:,:])+self_ene_in[im,:,:]

    #### hybridization function ####
    hyb=np.zeros((ndiv_tau,norb,norb),dtype=complex)
    hyb_tau=np.zeros((ndiv_tau+1,norb,norb),dtype=float)
    print "Computing Delta(omega_n)..."
    for im in range(ndiv_tau):
        hyb[im,:,:] = np.identity(norb)*(1J*matsubara_freq[im]+vmu)-invG0[im,:,:]-Hk_mean[:,:]
    print "Transforming Delta(omega_n) to Delta(tau)..."
    ft_to_tau_hyb(ndiv_tau,vbeta,matsubara_freq,tau,Hk_var,Hk_var2,hyb,hyb_tau,cutoff_fourie)

    f=open("hyb-n.txt","w")
    for im in range(ndiv_tau):
        print >>f, matsubara_freq[im], " ",
        for iorb in range(norb):
            for iorb2 in range(norb):
                print >>f, hyb[im,iorb,iorb2].imag, " ",
        print >>f
    f.close() 

    f=open("hyb-tau.txt","w")
    for itau in range(ndiv_tau+1):
        print >>f, tau[itau], " ",
        for iorb in range(norb):
            for iorb2 in range(norb):
                print >>f, hyb_tau[itau,iorb,iorb2].real, " ",
        print >>f
    f.close() 

    sys.stdout.flush()

    #### impurity solver ####
    imp_result,obs_meas = call_hyb_matrix(app_parms, tau, hyb_tau, hyb, rotmat_hyb, vmu)

    print "orbital_occ = ", imp_result["n"]
    print "tot_elec = ", np.sum(imp_result["n"])

    g_tau_im = np.zeros((ndiv_tau+1,norb,norb),dtype=float)
    G_imp = np.zeros((ndiv_tau,norb,norb),dtype=complex)
    for iorb in range(norb):
        for iorb2 in range(norb):
            g_tau_im[:,iorb,iorb2] = 0.5*(imp_result["Greens_imag_tau"][:,2*iorb,2*iorb2]+imp_result["Greens_imag_tau"][:,2*iorb+1,2*iorb2+1])
            G_imp[:,iorb,iorb2] = ft.to_freq_fermionic_real_field(ndiv_tau, vbeta, ndiv_tau, g_tau_im[:,iorb,iorb2], cutoff_fourie)

    #new self energy
    self_ene_out = np.zeros_like(self_ene_in)
    for im in range(ndiv_tau):
        self_ene_out[im,:,:]=np.identity(norb)*(1J*matsubara_freq[im]+vmu)-hyb[im,:,:]-Hk_mean[:,:]-inv(G_imp[im,:,:]) 
    np.save(app_parms["prefix"]+"-self_ene",self_ene_out)
    #np.savetxt(app_parms["prefix"]+"-self_ene_in_im11-isc"+str(isc)+".txt",self_ene_in[:,1,1].imag)
    #np.savetxt(app_parms["prefix"]+"-self_ene_im11-isc"+str(isc)+".txt",self_ene_out[:,1,1].imag)

    F = self_ene_out-self_ene_in
    for iorb in range(norb):
        for iorb2 in range(norb):
            F[:,iorb,iorb2] = F[:,iorb,iorb2]*dmp_fact
    #np.save("res-isc"+str(isc),F)

    print "max_diff", (np.absolute(F)).max()
    max_Gdiff =  (np.absolute(G_imp-G_latt)).max()
    print "max_Gdiff", max_Gdiff
    if 'TOLERANCE_G' in app_parms and max_Gdiff<app_parms['TOLERANCE_G']:
        F *= 0.0

    #update chemical potential
    if 'OPT_MU' in app_parms and app_parms['OPT_MU'] < 0:
        tote_imp = -1.0
        if 'n_rotated' in imp_result:
            print "Using n_rotated..."
            tote_imp = np.sum(imp_result["n_rotated"])
        else:
            print "Using n..."
            tote_imp = np.sum(imp_result["n"])

        print "tote_imp = ", tote_imp
        vmu = vmu-np.abs(app_parms['OPT_MU'])*(tote_imp-2*app_parms["N_ELEC"])
        print "new mu = ", vmu

    #update results
    dmft_result.update(Hk_mean, Hk_var, Hk_var2, vmu, G_latt, G_imp, g_tau_im, self_ene_out, hyb, hyb_tau)
    for key in obs_meas.keys():
        dmft_result[key] = obs_meas[key]
    dmft_result_list.append(copy.deepcopy(dmft_result))
    dump_results_modmft(app_parms,dmft_result_list)

    sys.stdout.flush()
    isc += 1
    return F

self_ene_init=np.zeros((ndiv_tau,norb,norb),dtype=complex)
if 'SIGMA_INPUT' in app_parms:
    print 'Reading ', app_parms['SIGMA_INPUT'], '...'
    self_ene_init=np.load(app_parms["SIGMA_INPUT"])

dmp_fact = matsubara_freq**(-2)
self_ene_dmp_init = np.array(self_ene_init)
for iorb in range(norb):
    for iorb2 in range(norb):
        self_ene_dmp_init[:,iorb,iorb2] = self_ene_dmp_init[:,iorb,iorb2]*dmp_fact

#sciopt.root(calc_diff,self_ene_init,method="anderson",options={'nit' : app_parms["MAX_IT"], 'fatol' : app_parms["CONVERGED"], 'disp': True, 'M': 10})
dmft_result = DMFTResult()
dmft_result_list = []
#sciopt.anderson(calc_diff,self_ene_dmp_init,iter=app_parms["MAX_IT"], f_rtol=app_parms["CONVERGED"], verbose=True)
mix = 0.5
if 'mix' in app_parms:
    mix = app_parms['mix']
sciopt.linearmixing(calc_diff,self_ene_dmp_init,alpha=mix,iter=app_parms["MAX_IT"],f_rtol=app_parms["CONVERGED"],line_search=None,verbose=True)
