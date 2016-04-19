import pyalps 
import pyalps.cthyb as cthyb # solver module
import pyalps.mpi as mpi     # MPI library
import numpy as np
import sys
import math
import time
import integration as intgr
import re 
import input_parms as inp
import os
import subprocess
import h5py
from h5dump import *
import fourier_transform as ft
from effective_int import *
import copy

class DMFTResult:
    def update(self, vmu, G_latt, G_imp, g_tau_im0, g_tau_im1, self_ene, W_imp, PI_imp, hyb, hyb_tau, U_retarded, nnt_0_0, nnt_1_0, nnt_1_1):
        self.vmu = vmu
        self.G_latt = np.array(G_latt)
        self.hyb = np.array(hyb)
        self.hyb_tau = np.array(hyb_tau)
        self.G_imp = np.array(G_imp)
        self.g_tau_im0 = np.array(g_tau_im0)
        self.g_tau_im1 = np.array(g_tau_im1)
        self.self_ene = np.array(self_ene)
        self.W_imp = np.array(W_imp)
        self.PI_imp = np.array(PI_imp)
        self.U_retarded = np.array(U_retarded)
        self.nnt_0_0 = nnt_0_0
        self.nnt_1_0 = nnt_1_0
        self.nnt_1_1 = nnt_1_1

def PH_symmetrize(data_tau):
    n=data_tau.size
    for i in range(n):
        data_tau[i]=0.5*(data_tau[i]+data_tau[n-1-i])
        data_tau[n-1-i]=data_tau[i]

def calc_retarded_int_with_corr(ndiv_tau,tau,matsubara_freq_boson,beta,Dn,Ktau,Kptau,Komega,cutoff):
    epsilon=1e-6

    for im in range(1,ndiv_tau):
        print im, (-Dn[im]/matsubara_freq_boson[im]**2).real, ((-1J*Dn[im])/matsubara_freq_boson[im]).real

    for it in range(ndiv_tau+1):
        rtmp=0.
        rtmp2=0.
        tau_tmp=tau[it]
        for im in range(1,cutoff):
            texp=np.exp(1J*matsubara_freq_boson[im]*tau_tmp)
            rtmp+=(Dn[im]*(1.0-texp)/matsubara_freq_boson[im]**2).real
            rtmp2+=((-1J*texp*Dn[im])/matsubara_freq_boson[im]).real
        Ktau[it]=2.0*rtmp/beta-(0.5*Dn[0]/beta)*tau_tmp*(beta-tau_tmp)
        Kptau[it]=2.0*rtmp2/beta+(Dn[0]/beta)*(tau_tmp-beta/2)

    Ktau[0]=0.0
    Ktau[ndiv_tau]=0.0
    #for it in range(1,ndiv_tau):
        #if Ktau[it]<epsilon:
            #Ktau[it]=epsilon
            #Kptau[it]=0.0

    Komega[0]=0.0
    for im in range(1,ndiv_tau):
        Komega[im]=(Dn[0]-Dn[im])/matsubara_freq_boson[im]**2
        Komega[0]-=Komega[im]

    #NaN check
    for im in range(ndiv_tau):
        if (math.isnan(Komega[im])):
            Komega[im] = 0.0
    for im in range(ndiv_tau+1):
        if (math.isnan(Ktau[im]) or math.isnan(Kptau[im])):
            Ktau[im] = 0.0
            Kptau[im] = 0.0

def calc_retarded_int(ndiv_tau,tau,matsubara_freq_boson,beta,Dn,Ktau,Kptau,Komega,cutoff):
    for it in range(ndiv_tau+1):
        rtmp=0.
        rtmp2=0.
        tau_tmp=tau[it]
        for im in range(1,cutoff):
            texp=np.exp(1J*matsubara_freq_boson[im]*tau_tmp)
            rtmp+=(-(texp-1.0)*(Dn[im]-Dn[0])/matsubara_freq_boson[im]**2).real
            rtmp2+=(-1J*texp*(Dn[im]-Dn[0])/matsubara_freq_boson[im]).real
        Ktau[it]=2.0*rtmp/beta
        Kptau[it]=2.0*rtmp2/beta

    Ktau_ext = np.array([-Ktau[1]]+Ktau.tolist()+[-Ktau[ndiv_tau-1]])
    #fit = interpolate.InterpolatedUnivariateSpline(np.linspace(0,beta,ndiv_tau),Ktau)
    Kptau[:] = (np.gradient(Ktau_ext,beta/ndiv_tau))[1:ndiv_tau+2]

    Komega[0]=0.0
    for im in range(1,ndiv_tau):
        Komega[im]=(Dn[0]-Dn[im])/matsubara_freq_boson[im]**2
        Komega[0]-=Komega[im]

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

def ft_to_tau_K(ndiv_tau, beta, matsubara_freq_boson, tau, data_n, data_tau, cutoff):
 for it in range(ndiv_tau+1):
     tau_tmp=tau[it]
     if it==0:
         tau_tmp=1E-4*(beta/ndiv_tau)
     if it==ndiv_tau:
         tau_tmp=beta-1E-4*(beta/ndiv_tau)

     ztmp=data_n[0]
     for im in range(1,cutoff):
         ztmp+=2.*data_n[im]*np.exp(-1J*matsubara_freq_boson[im]*tau_tmp)
     data_tau[it]=ztmp.real/beta

def ft_to_n_K(ndiv_tau, beta, matsubara_freq_boson, tau, data_tau, data_n):
 for im in range(ndiv_tau):
     tint=data_tau[:]*np.exp(1J*matsubara_freq_boson[im]*tau[:])
     data_n[im]=np.trapz(tint,tau)

def ft_to_tau_hyb(ndiv_tau, beta, matsubara_freq, tau, Vek, data_n, data_tau, cutoff):
 for it in range(ndiv_tau+1):
     tau_tmp=tau[it]
     if it==0:
         tau_tmp=1E-4*(beta/ndiv_tau)
     if it==ndiv_tau:
         tau_tmp=beta-1E-4*(beta/ndiv_tau)
     ztmp=0.0
     for im in range(cutoff):
         ztmp+=(data_n[im]+1J*Vek/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
     ztmp=ztmp/beta
     data_tau[it]=2.0*ztmp.real-0.5*Vek

#### setup parameters ####
app_parms = inp.read_input_parms(sys.argv[1]+'.h5')
app_parms['prefix'] = sys.argv[1]

ndiv_spectrum=app_parms["N_GRID_DOS"]
n_ivk_dos=app_parms["N_GRID_IVK_DOS"] 
PH=False
if "PH" in app_parms:
    if app_parms["PH"] != 0:
        PH=True
PM=True
vmix=1.0
nsc=app_parms["MAX_IT"]
vbeta=app_parms["BETA"]
vconverged=app_parms["CONVERGED"]
ndiv_tau=app_parms["NMATSUBARA"]
ndiv_tau_nn=app_parms['N_TAU_NN_MEAS']

#if ndiv_tau<5*vU*vbeta:
    #if mpi.rank==0:
        #print "The number of Matsubara frequencies should be larger than 5*beta*U!"

G0_real=np.zeros((ndiv_spectrum,),dtype=complex)
matsubara_freq=np.zeros((ndiv_tau,),dtype=float)
matsubara_freq_boson=np.zeros((ndiv_tau,),dtype=float)
tau=np.zeros((ndiv_tau+1,),dtype=float)
tau_mesh_nn=np.zeros((app_parms['N_TAU_NN_MEAS']+1,),dtype=float)

self_ene=np.zeros((2,ndiv_tau,),dtype=complex)
G_latt=np.zeros_like(self_ene,dtype=complex)
G_imp=np.zeros_like(self_ene,dtype=complex)
G0=np.zeros_like(self_ene,dtype=complex)
hyb=np.zeros((2,ndiv_tau,),dtype=complex)
hyb_tau=np.zeros((2,ndiv_tau+1,),dtype=float)

W_loc=np.zeros((ndiv_tau,),dtype=complex)
W_imp=np.zeros((ndiv_tau,),dtype=complex)
PI_imp=np.zeros((ndiv_tau,),dtype=complex)
U_retarded=np.zeros((ndiv_tau,),dtype=complex)
D_retarded=np.zeros((ndiv_tau,),dtype=complex)
K_omega=np.zeros((ndiv_tau,),dtype=complex)
K_omega_corr=np.zeros((ndiv_tau,),dtype=complex)
Kp_omega=np.zeros((ndiv_tau,),dtype=complex)
K_tau=np.zeros((ndiv_tau+1,),dtype=float)
Kp_tau=np.zeros((ndiv_tau+1,),dtype=float)
D_tau=np.zeros((ndiv_tau+1,),dtype=float)

# uncorrelated lattice Green function
for im in range(ndiv_tau):
    matsubara_freq[im]=((2*im+1)*np.pi)/vbeta
    matsubara_freq_boson[im]=((2*im)*np.pi)/vbeta

for it in range(ndiv_tau+1):
    tau[it]=(vbeta/ndiv_tau)*it
for it in range(ndiv_tau_nn+1):
    tau_mesh_nn[it]=(vbeta/ndiv_tau_nn)*it

#Loading interactions
ivk_ene,ivk_dos,ZB = inp.load_ivk_dos(app_parms)

#np.savetxt("ivk_ene", ivk_ene)
#np.savetxt("ivk_dos", ivk_dos)
#sys.exit(1)

#Initialization of DOS
e_smp=np.zeros((ndiv_spectrum,),dtype=float)
dos_smp=np.zeros((ndiv_spectrum,),dtype=float)
f=open(app_parms['DOSFILE'],"r")
for i in range(ndiv_spectrum):
    l = f.readline()
    data = re.split('[\s\t)(,]+', l.strip())
    e_smp[i]=float(data[0])
    dos_smp[i]=float(data[1])
if ZB != 1.0:
    print "Renormalizing band width using ZB=", ZB
    e_smp *= ZB
    dos_smp /= ZB
ek_mean = intgr.TrapezoidalRule2(ndiv_spectrum,e_smp,dos_smp*e_smp)
ek_var = intgr.TrapezoidalRule2(ndiv_spectrum,e_smp,dos_smp*(e_smp**2))

if mpi.rank==0:
    print "ek_mean= ", ek_mean
    print "ek_var= ", ek_var
    print "norm of DOS=", intgr.TrapezoidalRule2(ndiv_spectrum,e_smp,dos_smp)
    np.save(app_parms["prefix"]+"-ivk_ene",ivk_ene)
    np.save(app_parms["prefix"]+"-ivk_dos",ivk_dos)
vU=app_parms["U"]
vmu=app_parms["MU"]+vU/2
print "Using U,mu=", vU, vmu

#### self-consistent loop ####
dmft_result = DMFTResult()
dmft_result_list = []
for isc in range(nsc):
    if mpi.rank==0:
        print '\n'
        print 'self-consistent loop = ', isc

    if isc==0:
        if 'SIGMA_INPUT' in app_parms:
            print 'Reading ', app_parms['SIGMA_INPUT'], '...'
            #self_ene=read_text_data(app_parms["SIGMA_INPUT"],ndiv_tau,2,complex,1).transpose()
            self_ene=np.load(app_parms["SIGMA_INPUT"])
            
        if 'PI_INPUT' in app_parms:
            print 'Reading ', app_parms['PI_INPUT'], '...'
            #PI_imp=read_text_data(app_parms['PI_INPUT'],ndiv_tau,1,complex,2)[:,0]
            PI_itmp=np.load(app_parms['PI_INPUT'])

    if mpi.rank==0:
        f=open(app_parms['prefix']+'-self_ene-tmp.dat','w')
        for i in range(ndiv_tau):
            f.write("%i %f %f %f %f\n"%(i, self_ene[0][i].real, self_ene[0][i].imag, self_ene[1][i].real, self_ene[1][i].imag))
        f=open(app_parms['prefix']+'-PI-tmp.dat','w')
        for i in range(ndiv_tau):
            f.write("%i %f  %f %f\n"%(i, matsubara_freq_boson[i], PI_imp[i].real, PI_imp[i].imag))

    #### Lattice Green function ####
    for isp in range(2):
        for im in range(ndiv_tau):
            f_tmp = dos_smp/(1J*matsubara_freq[im]+vmu-e_smp-self_ene[isp][im])
            G_latt[isp][im]=intgr.TrapezoidalRule2(ndiv_spectrum,e_smp,f_tmp)
    for im in range(ndiv_tau):
        f_tmp = np.zeros((n_ivk_dos,),dtype=complex)
        #print "debug ", im, np.var(ivk_dos[im,:])
        for j in range(n_ivk_dos):
            if ivk_ene[im,j]-PI_imp[im] != 0:
                f_tmp[j] = ivk_dos[im,j]/(ivk_ene[im,j]-PI_imp[im])
            else:
                print "Caution: ivk_ene[j]-PI_imp[im]==0 at j,im=", j, im
                f_tmp[j] = 0.0
        W_loc[im]=intgr.TrapezoidalRule2(n_ivk_dos,ivk_ene[im],f_tmp)

    if mpi.rank==0:
        f=open(app_parms['prefix']+"-G_latt_n-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f %f %f\n"%(i, matsubara_freq[i], G_latt[0][i].real, G_latt[0][i].imag, G_latt[1][i].real, G_latt[1][i].imag))
        f.close()

        f=open(app_parms['prefix']+"-W_loc-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f\n"%(i, matsubara_freq_boson[i], W_loc[i].real, W_loc[i].imag))

    #### Cavity Green's function ####
    #G0=vmix*(1.0/(1.0/G_latt+self_ene))+(1.-vmix)*G0
    G0=1.0/(1.0/G_latt+self_ene)
    U_retarded=1./(1./W_loc+PI_imp)

    if mpi.rank==0:
        f=open(app_parms['prefix']+"-G0-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f %f %f\n"%(i, matsubara_freq[i], G0[0][i].real, G0[0][i].imag, G0[1][i].real, G0[1][i].imag))
        f.close()

        f=open(app_parms['prefix']+"-Ur-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f\n"%(i, matsubara_freq_boson[i], U_retarded[i].real, U_retarded[i].imag))
        f.close()

    # hybridization function
    hyb=1J*matsubara_freq+vmu-1.0/G0-ek_mean
    for isp in range(2):
        ft_to_tau_hyb(ndiv_tau,vbeta,matsubara_freq,tau,ek_var,hyb[isp],hyb_tau[isp],app_parms["CUTOFF_N_HYB"])
    if PH:
        for isp in range(2):
            for i in range(ndiv_tau):
                rtmp = 0.5*(hyb_tau[isp][i]+hyb_tau[isp][ndiv_tau-i])
                hyb_tau[isp][i] = rtmp
                hyb_tau[isp][ndiv_tau-i] = rtmp

    if mpi.rank==0:
        f=open(app_parms['prefix']+"-delta_n-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f %f %f\n"%(i, matsubara_freq[i], hyb[0][i].real, hyb[0][i].imag, hyb[1][i].real, hyb[1][i].imag))
        f.close()
    
        f=open(app_parms['prefix']+"-delta-raw-"+str(isc)+".dat","w")
        for i in range(ndiv_tau+1):
            f.write("%i %f %f\n"%(i,  hyb_tau[0][i], hyb_tau[1][i]))
        f.close()
   
    #### retarted U ####
    D_retarded=U_retarded-vU
    ft_to_tau_K(ndiv_tau, vbeta, matsubara_freq_boson, tau, D_retarded, D_tau, ndiv_tau/4)
    if mpi.rank==0:
        f=open(app_parms['prefix']+"-D-omega"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f\n"%(i, D_retarded[i].real, D_retarded[i].imag))
        f.close()
    #for i in range(ndiv_tau):
        #D_retarded[i]=min(D_retarded[i],0.0)

    ##### K(tau),Kp(tau) ####
    calc_retarded_int_with_corr(ndiv_tau,tau,matsubara_freq_boson,vbeta,D_retarded,K_tau,Kp_tau,K_omega,app_parms["CUTOFF_N_HYB"])
    #calc_retarded_int(ndiv_tau,tau,matsubara_freq_boson,vbeta,D_retarded,K_tau,Kp_tau,K_omega,app_parms["CUTOFF_N_HYB"])

    if mpi.rank==0:
        f=open(app_parms['prefix']+"-K-omega"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f\n"%(i, matsubara_freq_boson[i], K_omega[i].real))
        f.close()

    if mpi.rank==0:
        f=open(app_parms['prefix']+"-K-"+str(isc)+".dat","w")
        for i in range(ndiv_tau+1):
            f.write("%i %f %f\n"%(i, K_tau[i], Kp_tau[i]))
        f.close()

    #### impurity solver ####
    G_imp_old=G_imp.copy()
    W_imp_old=W_imp.copy()

    imp_parms={
    'SWEEPS'              : app_parms["SWEEPS"],
    'MAX_TIME'            : app_parms["MAX_TIME"],
    'THERMALIZATION'      : app_parms["THERMALIZATION"],
    'SEED'                : 0,
    'N_MEAS'              : app_parms["N_MEAS"],
    'N_HISTOGRAM_ORDERS'  : app_parms["N_ORDER"],
    'N_ORBITALS'          : 2,
    'U'                   : vU,
    'MU'                  : vmu-2*Kp_tau[0], #debug
    'DELTA'               : app_parms['prefix']+"-delta-"+str(isc)+".dat",
    'RET_INT_K'           : app_parms['prefix']+"-K-"+str(isc)+".dat",
    'N_TAU'               : ndiv_tau,
    'MEASURE_freq'        : 0,
    'N_MATSUBARA'         : ndiv_tau,
    #'N_W'                 : app_parms['NMATSUBARA_FREQ_MEAS'],
    'BETA'                : vbeta,
    'TEXT_OUTPUT'         : 1,
    'MEASURE_nnt'         : app_parms['N_MEAS_nnt'],
    'N_nn'                : app_parms['N_TAU_NN_MEAS'],
    'VERBOSE'           : 1,
    'OUTPUT_PERIOD'       : 1000
    }
    ierror=0
    for isp in range(2):
        for i in range(ndiv_tau+1):
            if hyb_tau[isp][i]>0:
                ierror+=1
                hyb_tau[isp][i]=0.0
    if mpi.rank==0:
        if ierror>0:
            print "Warning: hybridization function is positive!"
        f=open(app_parms['prefix']+"-delta-"+str(isc)+".dat","w")
        for i in range(ndiv_tau+1):
            f.write("%i %f %f\n"%(i,  hyb_tau[0][i], hyb_tau[1][i]))
        f.close()

    mpi.world.barrier()
    if app_parms['PY_MODULE_HYB']!=0:
        foutput='results.out.h5'
        print 'Calling the impurity solover as a python module...'
        cthyb.solve(imp_parms)
    else:
        print 'Calling the impurity solover as an external program...'
        foutput=app_parms['prefix']+'-input_hyb.out.h5'
        f = h5py.File(app_parms['prefix']+'-input_hyb.h5', 'w')
        subgroup = f.create_group('parameters')
        for k,v in imp_parms.iteritems():
            if isinstance(v,int):
                dset = subgroup.create_dataset(k,data=v,dtype='i4') #hybridization does not accept 64-bit integers.
            else:
                dset = subgroup.create_dataset(k, data=v)
        f.close()
        cmd=app_parms['CMD_MPI']+' '+str(app_parms['N_MPI_PROCESS'])+' '+str(app_parms['HYB_PATH'])+' '+app_parms['prefix']+'-input_hyb.h5 > output-hyb-'+str(isc)
        print cmd
        os.system(cmd)
        #subprocess.call(cmd, shell=True)
        print "Finished hybridization"

    mpi.world.barrier()

    ll=pyalps.load.Hdf5Loader()
    listobs=["g_0", "g_1", "nnt_0_0", "nnt_1_0", "nnt_1_1", "density_0", "density_1"]
    data=ll.ReadMeasurementFromFile(pyalps.getResultFiles(pattern=foutput), respath='/simulation/results', measurements=listobs, verbose=True)
    data_flatten=pyalps.flatten(data)
    g_tau_im0 = data_flatten[0].y.mean
    g_tau_im1 = data_flatten[1].y.mean
    g_tau_im0[0] *= 2.
    g_tau_im1[0] *= 2.
    g_tau_im0[ndiv_tau] *= 2.
    g_tau_im1[ndiv_tau] *= 2.

    occ=float(data_flatten[5].y[0].mean+data_flatten[6].y[0].mean)
    print "tot_occ=",occ
    if 'OPT_MU' in app_parms and 'N_ELEC' in app_parms: 
        vmu = vmu-app_parms['OPT_MU']*(occ-2*app_parms["N_ELEC"])
        print "new vmu ", vmu

    #note: contribution from the connected diagram should substracted.
    chi_loc = np.zeros((ndiv_tau,),dtype=complex)
    #chi_loc[0:app_parms['NMATSUBARA_FREQ_MEAS']] = data_flatten[13].y.mean+2.*data_flatten[14].y.mean+data_flatten[15].y.mean
    #chi_loc[0] -= vbeta*(occ**2)

    nnt_0_0 = data_flatten[2].y.mean
    nnt_1_0 = data_flatten[3].y.mean
    nnt_1_1 = data_flatten[4].y.mean

    chi_loc_t = data_flatten[2].y.mean+2*data_flatten[3].y.mean+data_flatten[4].y.mean
    chi_loc00_t = data_flatten[2].y.mean
    chi_loc10_t = data_flatten[3].y.mean
    chi_loc11_t = data_flatten[4].y.mean
    if PH:
        PH_symmetrize(chi_loc_t)
    chi_loc_ft = ft.to_freq_bosonic_real_field(app_parms['N_TAU_NN_MEAS'],vbeta,ndiv_tau,chi_loc_t,app_parms['N_TAU_NN_MEAS'])
    chi_loc_ft[0]-= vbeta*(occ**2)

    if PH:
        PH_symmetrize(g_tau_im0)
        PH_symmetrize(g_tau_im1)
    G_imp[0,:] = ft.to_freq_fermionic_real_field(ndiv_tau, vbeta, ndiv_tau, g_tau_im0, ndiv_tau/2)
    G_imp[1,:] = ft.to_freq_fermionic_real_field(ndiv_tau, vbeta, ndiv_tau, g_tau_im1, ndiv_tau/2)
    if PM:
        G_imp[0]=0.5*(G_imp[0]+G_imp[1])
        G_imp[1]=1.*(G_imp[0])
    if PH:
        for i in range(2):
            G_imp[i]=1J*(G_imp[i].imag)

    if mpi.rank==0:
        f=open(app_parms['prefix']+"-g_imp-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f  %f %f     %f %f\n"%(i, matsubara_freq[i], G_imp[0][i].real, G_imp[0][i].imag, G_imp[1][i].real, G_imp[1][i].imag))
        f.close()

        f=open(app_parms['prefix']+"-g_imp_tau-"+str(isc)+".dat","w")
        for i in range(ndiv_tau+1):
            f.write("%i %f %f\n"%(i, g_tau_im0[i], g_tau_im1[i]))
        f.close()

        f=open(app_parms['prefix']+"-chi_imp-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f  %f %f\n"%(i, matsubara_freq_boson[i], chi_loc[i].real, chi_loc_ft[i].real))
        f.close()

        f=open(app_parms['prefix']+"-chi_imp_t-"+str(isc)+".dat","w")
        for i in range(app_parms['N_TAU_NN_MEAS']+1):
            f.write("%i   %f %f %f %f\n"%(i, chi_loc_t[i], chi_loc00_t[i], chi_loc10_t[i], chi_loc11_t[i]))
        f.close()


    chi_loc[:]=1.0*chi_loc_ft[:]
    if PH:
        chi_loc[:]=chi_loc[:].real

    W_imp=U_retarded-chi_loc*U_retarded**2
    if mpi.rank==0:
        f=open(app_parms['prefix']+"-W_imp-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f  %f %f\n"%(i, matsubara_freq_boson[i], W_imp[i].real, W_imp[i].imag))
        f.close()

    #convergence check
    maxdiff_Gimp=(np.absolute(G_imp-G_imp_old)).max()
    maxdiff_Wimp=(np.absolute(W_imp-W_imp_old)).max()
    if mpi.rank==0:
        print " maxdiff_Gimp= ", maxdiff_Gimp
        print " maxdiff_Wimp= ", maxdiff_Wimp
    if (maxdiff_Gimp<vconverged):
        if mpi.rank==0:
            print "g_imp converged"
        break

    #update self energy and PI
    self_ene_old = np.array(self_ene)
    PI_imp_old = np.array(PI_imp)
    for i in range(2):
        self_ene[i]=1J*matsubara_freq+vmu-hyb[i]-ek_mean-1/G_imp[i]
    if PM:
        self_ene[0] = 0.5*(self_ene[0]+self_ene[1])
        self_ene[1] = 1.0*self_ene[0]
    PI_imp=1/U_retarded-1/W_imp
    mix = 0.5
    if app_parms.has_key('MIXING'):
    	mix = app_parms['MIXING']
    self_ene = mix*self_ene+(1.0-mix)*self_ene_old
    PI_imp = mix*PI_imp+(1.0-mix)*PI_imp_old

    if mpi.rank==0:
        f=open(app_parms['prefix']+"-self_ene-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f %f\n"%(i, self_ene[0][i].real, self_ene[0][i].imag, self_ene[1][i].real, self_ene[1][i].imag))
        f.close()

        f=open(app_parms['prefix']+"-PI-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f\n"%(i, matsubara_freq_boson[i], PI_imp[i].real, PI_imp[i].imag))
        f.close()

    #debug PI should be real and not larger than 0.
    if app_parms.has_key('NON_POSITIVE_PI'):
        if app_parms['NON_POSITIVE_PI']==1:
            for i in range(ndiv_tau):
                PI_imp[i]=min(PI_imp[i].real,-1E-10)

    dmft_result.update(vmu, G_latt, G_imp, g_tau_im0, g_tau_im1, self_ene, W_imp, PI_imp, hyb, hyb_tau, U_retarded, nnt_0_0, nnt_1_0, nnt_1_1)
    dmft_result_list.append(copy.deepcopy(dmft_result))
    dump_results_modmft(app_parms,dmft_result_list)
    np.save(app_parms["prefix"]+"-self_ene",self_ene)
    np.save(app_parms["prefix"]+"-PI",PI_imp)

#f=open("G0.dat","w")
#f.write("%i\n"%(ndiv_tau))
#for i in range(ndiv_tau):
    #f.write("%i %f %f %f %f\n"%(i, G0[0][i].real, G0[0][i].imag, G0[1][i].real, G0[1][i].imag))
