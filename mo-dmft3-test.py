import pyalps 
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
from hyb_matrix import call_hyb_matrix
from h5dump import *
import copy
import h5py
import impurity_model
import time


#### setup parameters ####
app_parms = inp.read_input_parms(sys.argv[1]+'.h5')
app_parms['prefix'] = sys.argv[1]
h5f = h5py.File(sys.argv[1]+'.h5','r')
imp_model = impurity_model.OrbitalModel(h5f)
print "Hk_mean", imp_model.get_moment(1)
h5f.close()

norb=app_parms["N_ORB"]
nflavor=2*norb
PM=True
vmix=1.0
vmu=app_parms["MU"]
tote=app_parms["N_ELEC"]
vbeta=app_parms["BETA"]
vconverged=app_parms["CONVERGED"]
ndiv_tau=app_parms["NMATSUBARA"]
cutoff_fourie=app_parms["CUTOFF_FOURIE"]

matsubara_freq=np.zeros((ndiv_tau,),dtype=float)
tau=np.zeros((ndiv_tau+1,),dtype=float)

# uncorrelated lattice Green function
for im in range(ndiv_tau):
    matsubara_freq[im]=((2*im+1)*np.pi)/vbeta
for it in range(ndiv_tau+1):
    tau[it]=(vbeta/ndiv_tau)*it

#print "Hk_mean", imp_model.get_moment(1)

#Fourie transformer
fourie_transformer = ft.FourieTransformer(imp_model)
Hk_mean = 1.*imp_model.get_moment(1)
#print "Hk_mean", Hk_mean

#Rotation matrix of hybridization function
evals,evecs=eigh_ordered(fourie_transformer.M1_)
if app_parms['BASIS_ROT']==0:
    rotmat_hyb = np.identity(nflavor,dtype=complex)
else:
    rotmat_hyb = 1.0*evecs

if 'mu' in app_parms:
    raise RuntimeError("Do not use mu")

isc = 0

#### self-consistent loop ####
def calc_diff(self_ene_dmp_in):
    global isc, vmu

    #Rescale self_energy
    self_ene_in = np.array(self_ene_dmp_in)
    for iflavor in range(nflavor):
        for iflavor2 in range(nflavor):
            self_ene_in[:,iflavor,iflavor2] = self_ene_dmp_in[:,iflavor,iflavor2]/dmp_fact

    #### Lattice Green function ####
    print "Computing lattice Green's function..."
    for i_chem in range(5): #By default, we do it 5 times.
        time1 = time.clock()
        G_latt,tote_tmp = imp_model.calc_Glatt_new(vbeta,matsubara_freq,self_ene_in,vmu)
        time2 = time.clock()
        print "Computing G_latt(tau) took ", time2-time1, " sec."
        G_latt_tau,G_latt_tail,G_latt_rest = fourie_transformer.G_freq_to_tau(G_latt,ndiv_tau,vbeta,cutoff_fourie)

        if 'OPT_MU' in app_parms and app_parms['OPT_MU'] != 0:
            ntot = 0.0
            for ie in range(nflavor):
                ntot += -G_latt_tau[-1,ie,ie].real
            vmu = (vmu-app_parms['OPT_MU']*(ntot-2*app_parms["N_ELEC"])).real
            print "tot_Glatt = ", ntot
            print "new mu = ", vmu
        else:
            break

    np.save(app_parms["prefix"]+"-G_latt", G_latt)
    np.save(app_parms["prefix"]+"-G_latt_tau", G_latt_tau)
    np.save(app_parms["prefix"]+"-G_latt_tail", G_latt_tail)
    np.save(app_parms["prefix"]+"-G_latt_rest", G_latt_rest)
    assert np.amax(np.abs(G_latt-G_latt_tail-G_latt_rest))<1e-10

    #### Cavity Green's function ####
    print "Computing cavity Green's function..."
    invG0 = np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
    for im in range(ndiv_tau):
        invG0[im,:,:]=inv(G_latt[im,:,:])+self_ene_in[im,:,:]

    #### Hybridization function ####
    hyb=np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
    print "Computing Delta(omega_n)..."
    for im in range(ndiv_tau):
        hyb[im,:,:] = np.identity(nflavor)*(1J*matsubara_freq[im]+vmu)-invG0[im,:,:]-Hk_mean[:,:]
    print "Transforming Delta(omega_n) to Delta(tau)..."
    hyb_tau,high_freq_tail,hyb_rest = fourie_transformer.hyb_freq_to_tau(hyb,ndiv_tau,vbeta,cutoff_fourie)
    np.save(app_parms["prefix"]+"-hyb", hyb)
    np.save(app_parms["prefix"]+"-hyb_tau", hyb_tau)
    np.save(app_parms["prefix"]+"-hyb_tail", high_freq_tail)
    np.save(app_parms["prefix"]+"-hyb_rest", hyb_rest)

    f=open("hyb-n.txt","w")
    for im in range(ndiv_tau):
        print >>f, matsubara_freq[im], " ",
        for iflavor in range(nflavor):
            for iflavor2 in range(nflavor):
                print >>f, hyb[im,iflavor,iflavor2].real, " ", hyb[im,iflavor,iflavor2].imag, " ",
                print >>f, high_freq_tail[im,iflavor,iflavor2].real, " ", high_freq_tail[im,iflavor,iflavor2].imag, " ",
                print >>f, hyb_rest[im,iflavor,iflavor2].real, " ", hyb_rest[im,iflavor,iflavor2].imag, "    ",
        print >>f
    f.close() 

    f=open("hyb-tau.txt","w")
    for itau in range(ndiv_tau+1):
        print >>f, tau[itau], " ",
        for iflavor in range(nflavor):
            for iflavor2 in range(nflavor):
                print >>f, hyb_tau[itau,iflavor,iflavor2].real, " ", hyb_tau[itau,iflavor,iflavor2].imag, " ",
        print >>f
    f.close() 

    sys.stdout.flush()

    #### Impurity solver ####
    imp_result,obs_meas = call_hyb_matrix(app_parms, imp_model, tau, hyb_tau, hyb, rotmat_hyb, vmu)

    print "orbital_occ = ", imp_result["n"]
    print "tot_elec = ", np.sum(imp_result["n"])

    g_tau_im = np.zeros((ndiv_tau+1,nflavor,nflavor),dtype=complex)
    G_imp = np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
    for iflavor in range(nflavor):
        for iflavor2 in range(nflavor):
            g_tau_im[:,iflavor,iflavor2] = imp_result["Greens_imag_tau"][:,iflavor,iflavor2]
            G_imp[:,iflavor,iflavor2] = ft.G_tau_to_freq(ndiv_tau, vbeta, g_tau_im[:,iflavor,iflavor2])

    np.save(app_parms["prefix"]+"-G_tau",g_tau_im)
    np.save(app_parms["prefix"]+"-G_imp",G_imp)

    #Update self energy
    self_ene_out = np.zeros_like(self_ene_in)
    for im in range(ndiv_tau):
        self_ene_out[im,:,:]=np.identity(nflavor)*(1J*matsubara_freq[im]+vmu)-hyb[im,:,:]-Hk_mean[:,:]-inv(G_imp[im,:,:]) 
    np.save(app_parms["prefix"]+"-self_ene",self_ene_out)
    np.savetxt(app_parms["prefix"]+"-self_ene_in_im11-isc"+str(isc)+".txt",self_ene_in[:,1,1].imag)
    np.savetxt(app_parms["prefix"]+"-self_ene_im11-isc"+str(isc)+".txt",self_ene_out[:,1,1].imag)

    F = self_ene_out-self_ene_in
    for iflavor in range(nflavor):
        for iflavor2 in range(nflavor):
            F[:,iflavor,iflavor2] = F[:,iflavor,iflavor2]*dmp_fact

    print "max_diff", (np.absolute(F)).max()
    max_Gdiff =  (np.absolute(G_imp-G_latt)).max()
    print "max_Gdiff", max_Gdiff
    if 'TOLERANCE_G' in app_parms and max_Gdiff<app_parms['TOLERANCE_G']:
        F *= 0.0

    #Update chemical potential
    #if 'OPT_MU' in app_parms and app_parms['OPT_MU'] != 0:
        #tote_imp = np.sum(imp_result["n"])
        #print "tote_imp = ", tote_imp
        #vmu = (vmu-app_parms['OPT_MU']*(tote_imp-2*app_parms["N_ELEC"])).real
        #print "new mu = ", vmu

    #Update results
    dmft_result.update(imp_model.get_moment(1), imp_model.get_moment(2), imp_model.get_moment(3), vmu, G_latt, G_imp, g_tau_im, self_ene_out, hyb, hyb_tau)
    for key in obs_meas.keys():
        dmft_result[key] = obs_meas[key]
    dmft_result_list.append(copy.deepcopy(dmft_result))
    dump_results_modmft(app_parms,dmft_result_list)

    sys.stdout.flush()
    isc += 1
    return F

self_ene_init=np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
if 'SIGMA_INPUT' in app_parms:
    print 'Reading ', app_parms['SIGMA_INPUT'], '...'
    self_ene_init=np.load(app_parms["SIGMA_INPUT"])

dmp_fact = matsubara_freq**(-2)
self_ene_dmp_init = np.array(self_ene_init)
for iflavor in range(nflavor):
    for iflavor2 in range(nflavor):
        self_ene_dmp_init[:,iflavor,iflavor2] = self_ene_dmp_init[:,iflavor,iflavor2]*dmp_fact

#sciopt.root(calc_diff,self_ene_init,method="anderson",options={'nit' : app_parms["MAX_IT"], 'fatol' : app_parms["CONVERGED"], 'disp': True, 'M': 10})
dmft_result = DMFTResult()
dmft_result_list = []
mix = 0.5
if 'mix' in app_parms:
    mix = app_parms['mix']
sciopt.linearmixing(calc_diff,self_ene_dmp_init,alpha=mix,iter=app_parms["MAX_IT"],f_rtol=app_parms["CONVERGED"],line_search=None,verbose=True)
#sciopt.anderson(calc_diff,self_ene_dmp_init,iter=app_parms["MAX_IT"], f_rtol=app_parms["CONVERGED"], verbose=True)
