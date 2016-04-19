import pyalps 
import pyalps.cthyb as cthyb # solver module
import pyalps.mpi as mpi     # MPI library
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
import h5py
import fourier_transform as ft
from lib import *

from hyb import call_hyb_matrix

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

def ft_to_tau_hyb(ndiv_tau, beta, matsubara_freq, tau, Vhk, data_n, data_tau, cutoff):
 norb = Vhk.shape[0]
 for it in range(ndiv_tau+1):
     tau_tmp=tau[it]
     if it==0:
         tau_tmp=1E-4*(beta/ndiv_tau)
     if it==ndiv_tau:
         tau_tmp=beta-1E-4*(beta/ndiv_tau)
     ztmp=np.zeros((norb,norb),dtype=complex)
     for im in range(cutoff):
         ztmp+=(data_n[im]+1J*Vhk/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
     ztmp=ztmp/beta
     data_tau[it]=2.0*ztmp.real-0.5*Vhk

def calc_Glatt(beta,matsubara_freq,Hk,self_ene,vmu):
    ndiv_tau = len(matsubara_freq)
    norb = self_ene.shape[1]
    G_latt=np.zeros((ndiv_tau,norb,norb),dtype=complex)
    for im in range(ndiv_tau):
        G_latt[im,:,:] = 0.0+0.0J
        digmat = (1J*matsubara_freq[im]+vmu)*np.identity(norb)
        for ik in range(Nk):
            G_latt[im,:,:] += inv(digmat-Hk[ik,:,:]-self_ene[im,:,:])
        G_latt[im,:,:] /= Nk
        print im, G_latt[im,0,0].imag

    tau_tmp = beta-1E-4*(beta/ndiv_tau)
    ztmp = 0.0
    cutoff = ndiv_tau/2
    for im in range(cutoff):
        for iorb in range(norb):
            ztmp += (G_latt[im,iorb,iorb]+1J/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
    rtmp = 2.0*ztmp.real/beta-0.5*norb
    
    return G_latt,-rtmp

#### setup parameters ####
app_parms = inp.read_input_parms(sys.argv[1]+'.h5')

norb=app_parms["N_ORB"]
nkdiv=app_parms["N_K_DIV"]
#Nk=app_parms["N_K"]
Nk=nkdiv**3 #3D DOS
PM=True
vmix=1.0
nsc=app_parms["MAX_IT"]
vmu=app_parms["MU"]
tote=app_parms["N_ELEC"]
vbeta=app_parms["BETA"]
vconverged=app_parms["CONVERGED"]
ndiv_tau=app_parms["NMATSUBARA"]

#if ndiv_tau<5*vU*vbeta:
    #if mpi.rank==0:
        #print "The number of Matsubara frequencies should be larger than 5*beta*U!"

matsubara_freq=np.zeros((ndiv_tau,),dtype=float)
matsubara_freq_boson=np.zeros((ndiv_tau,),dtype=float)
tau=np.zeros((ndiv_tau+1,),dtype=float)

self_ene=np.zeros((ndiv_tau,norb,norb),dtype=complex)
G_latt=np.zeros_like(self_ene,dtype=complex)
hyb=np.zeros((ndiv_tau,norb,norb),dtype=complex)
hyb_tau=np.zeros((ndiv_tau+1,norb,norb),dtype=float)

# uncorrelated lattice Green function
for im in range(ndiv_tau):
    matsubara_freq[im]=((2*im+1)*np.pi)/vbeta
for it in range(ndiv_tau+1):
    tau[it]=(vbeta/ndiv_tau)*it

#Initialization of H(k)
H0 = np.zeros((norb,norb),dtype=float)#We assume these are real.
for iorb in range(norb):
    for iorb2 in range(norb):
        if iorb!=iorb2:
            H0[iorb,iorb2] = -app_parms["t'"]

for iorb in range(norb):
    H0[iorb,iorb] = app_parms["E"+str(iorb)]

Hnn = np.zeros((norb,norb),dtype=float)
for i in range(norb):
    Hnn[i,i] = -1.0

#DEBUG
#for iorb in range(norb):
    #for iorb2 in range(norb):
        #if iorb!=iorb2:
            #Hnn[iorb,iorb2] = -app_parms["t'"]

Hk = np.zeros((Nk,norb,norb),dtype=float)
Hk2 = np.zeros((Nk,norb,norb),dtype=float) #H(k)*H(k): not element-wise
ik = 0
for i in range(nkdiv):
    for j in range(nkdiv):
        for k in range(nkdiv):
            Hk[ik,:,:] = H0[:,:]+2.0*app_parms["t"]*(cos(2*i*pi/nkdiv)+cos(2*j*pi/nkdiv)+cos(2*k*pi/nkdiv))*Hnn[:,:]
            Hk2[ik,:,:] = np.dot(Hk[ik,:,:],Hk[ik,:,:])
            ik+=1

Hk_mean = np.zeros((norb,norb),dtype=float)
Hk_var = np.zeros((norb,norb),dtype=float)
for iorb in range(norb):
    for jorb in range(norb):
        Hk_mean[iorb,jorb] = np.mean(Hk[:,iorb,jorb])
        Hk_var[iorb,jorb] = np.mean(Hk2[:,iorb,jorb])
Hk_var[:,:] -= np.dot(Hk_mean,Hk_mean)
print "Hk_mean=", Hk_mean
print "Hk_var=", Hk_var

print_ek_info(Nk,norb,Hk)

if 'mu' in app_parms:
    raise RuntimeError("Do not use mu")

#### self-consistent loop ####
for isc in range(nsc):
    if mpi.rank==0:
        print 'self-consistent loop = ', isc

    if isc==0:
        if 'SIGMA_INPUT' in app_parms:
            print 'Reading ', app_parms['SIGMA_INPUT'], '...'
            self_ene=np.load(app_parms["SIGMA_INPUT"])
            
    #### Lattice Green function ####
    mu_max=app_parms["MU_MAX"]
    mu_min=app_parms["MU_MIN"]
    if (mu_max==mu_min and app_parms["MU_MAX"]==mu_max):
        vmu = mu_max
        G_latt,tote_tmp = calc_Glatt(vbeta,matsubara_freq,Hk,self_ene,vmu)
        print "tote evaluated by G_latt is ", tote_tmp
    else:
        tote_eps = 1E-3
        G_latt,tote_max = calc_Glatt(vbeta,matsubara_freq,Hk,self_ene,mu_max)
        print "tote_max ", tote_max
        if (tote_max <= tote):
            raise RuntimeError("MU_MAX is too small")
        G_latt,tote_min = calc_Glatt(vbeta,matsubara_freq,Hk,self_ene,mu_min)
        print "tote_min ", tote_min
        if (tote_min >= tote):
            raise RuntimeError("MU_MAX is too large")
        while True:
            vmu = ((tote_max-tote)*mu_min-(tote_min-tote)*mu_max)/(tote_max-tote_min)
            G_latt,tote_tmp = calc_Glatt(vbeta,matsubara_freq,Hk,self_ene,vmu)
            print "vmu,tote=", vmu, tote_tmp
            if (abs(tote_tmp-tote)<tote_eps):
                break
            if (tote_tmp>tote):
                mu_max = vmu
                tote_max = tote_tmp
            else:
                mu_min = vmu
                tote_min = tote_tmp
    print "Using mu=", vmu
    np.savetxt("G_latt00-isc"+str(isc)+".txt",G_latt[:,0,0].imag)
    np.savetxt("G_latt01-isc"+str(isc)+".txt",G_latt[:,0,1].imag)
    np.savetxt("G_latt11-isc"+str(isc)+".txt",G_latt[:,1,1].imag)
    np.savetxt("G_latt22-isc"+str(isc)+".txt",G_latt[:,2,2].imag)

    #### Cavity Green's function ####
    invG0 = np.zeros((ndiv_tau,norb,norb),dtype=complex)
    for im in range(ndiv_tau):
        print im
        invG0[im,:,:]=inv(G_latt[im,:,:])+self_ene[im,:,:]

    #### hybridization function ####
    hyb = np.zeros((ndiv_tau,norb,norb),dtype=complex)
    for im in range(ndiv_tau):
        hyb[im,:,:] = np.identity(norb)*(1J*matsubara_freq[im]+vmu)-invG0[im,:,:]-Hk_mean[:,:]
    ft_to_tau_hyb(ndiv_tau,vbeta,matsubara_freq,tau,Hk_var,hyb,hyb_tau,ndiv_tau/2)

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

    #### impurity solver ####
    imp_result = call_hyb_matrix(app_parms, tau, hyb_tau, vmu)

    print "orbital_occ = ", imp_result["n"]
    print "tot_elec = ", np.sum(imp_result["n"])

    g_tau_im = np.zeros((ndiv_tau+1,norb,norb),dtype=float)
    G_imp = np.zeros((ndiv_tau,norb,norb),dtype=complex)
    for iorb in range(norb):
        for iorb2 in range(norb):
            g_tau_im[:,iorb,iorb2] = 0.5*(imp_result["Greens_imag_tau"][:,2*iorb,2*iorb2]+imp_result["Greens_imag_tau"][:,2*iorb+1,2*iorb2+1])
            G_imp[:,iorb,iorb2] = ft.to_freq_fermionic_real_field(ndiv_tau, vbeta, ndiv_tau, g_tau_im[:,iorb,iorb2], ndiv_tau/2)
    #G_imp = 1J*(G_imp.imag)

    np.save("g_tau_im-isc"+str(isc),g_tau_im)
    np.save("G_imp-isc"+str(isc),G_imp.imag)

    np.savetxt("g_tau_im00-isc"+str(isc)+".txt",g_tau_im[:,0,0])
    np.savetxt("g_tau_im01-isc"+str(isc)+".txt",g_tau_im[:,0,1])
    np.savetxt("g_tau_im12-isc"+str(isc)+".txt",g_tau_im[:,1,2])
    np.savetxt("g_tau_im11-isc"+str(isc)+".txt",g_tau_im[:,1,1])
    np.savetxt("g_tau_im22-isc"+str(isc)+".txt",g_tau_im[:,2,2])

    np.savetxt("G_imp-isc"+str(isc)+"00.txt",G_imp[:,0,0].imag)
    np.savetxt("G_imp-isc"+str(isc)+"01.txt",G_imp[:,0,1].imag)
    np.savetxt("G_imp-isc"+str(isc)+"11.txt",G_imp[:,1,1].imag)
    np.savetxt("G_imp-isc"+str(isc)+"22.txt",G_imp[:,2,2].imag)

    #convergence check
    maxdiff_Gimp=(np.absolute(G_imp-G_latt)[0:min(100,ndiv_tau)]).max()
    print " maxdiff_Gimp= ", maxdiff_Gimp
    if (maxdiff_Gimp<vconverged):
        print "g_imp converged"
        break

    #update self energy and PI
    mix = 0.5
    self_ene_old = np.array(self_ene)
    for im in range(ndiv_tau):
        self_ene[im,:,:]=np.identity(norb)*(1J*matsubara_freq[im]+vmu)-hyb[im,:,:]-Hk_mean[:,:]-inv(G_imp[im,:,:])

    self_ene = mix*self_ene+(1.0-mix)*self_ene_old
    np.save("self_ene",self_ene)
