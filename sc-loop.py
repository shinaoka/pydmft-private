import pyalps 
import pyalps.cthyb as cthyb # solver module
import pyalps.mpi as mpi     # MPI library
import numpy as np
import sys
import time
import integration as intgr
import re 
import input_parms as inp
import os
import subprocess
import h5py
import fourier_transform as ft

def do_step(app_parms, )
    PM=True
    norb=app_parms["N_ORB"]
    vU=app_parms["U"]
    vmu=app_parms["MU"]+vU/2
    vbeta=app_parms["BETA"]
    vconverged=app_parms["CONVERGED"]
    ndiv_tau=app_parms["NMATSUBARA"]
    ndiv_tau_nn=app_parms['N_TAU_NN_MEAS']

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

    # uncorrelated lattice Green function
    for im in range(ndiv_tau):
        matsubara_freq[im]=((2*im+1)*np.pi)/vbeta
        matsubara_freq_boson[im]=((2*im)*np.pi)/vbeta

    for it in range(ndiv_tau+1):
        tau[it]=(vbeta/ndiv_tau)*it
    for it in range(ndiv_tau_nn+1):
        tau_mesh_nn[it]=(vbeta/ndiv_tau_nn)*it

    #Initialization of DOS
    e_smp=np.zeros((ndiv_spectrum,),dtype=float)
    dos_smp=np.zeros((ndiv_spectrum,),dtype=float)
    f=open(app_parms['DOSFILE'],"r")
    for i in range(ndiv_spectrum):
        l = f.readline()
        data = re.split('[\s\t)(,]+', l.strip())
        e_smp[i]=float(data[0])
        dos_smp[i]=float(data[1])
    ek_mean = intgr.TrapezoidalRule2(ndiv_spectrum,e_smp,dos_smp*e_smp)
    ek_var = intgr.TrapezoidalRule2(ndiv_spectrum,e_smp,dos_smp*(e_smp**2))

    ivk_ene,ivk_dos = inp.load_ivk_dos(app_parms)
    
    if mpi.rank==0:
        print "ek_mean= ", ek_mean
        print "ek_var= ", ek_var
        print "norm of DOS=", intgr.TrapezoidalRule2(ndiv_spectrum,e_smp,dos_smp)

    #### Lattice Green function ####
    for isp in range(2):
        for im in range(ndiv_tau):
            f_tmp = dos_smp/(1J*matsubara_freq[im]+vmu-e_smp-self_ene[isp][im])
            G_latt[isp][im]=intgr.TrapezoidalRule2(ndiv_spectrum,e_smp,f_tmp)

    #### Cavity Green's function ####
    G0=vmix*(1.0/(1.0/G_latt+self_ene))+(1.-vmix)*G0
    U_retarded=1./(1./W_loc+PI_imp)

    if mpi.rank==0:
        f=open("G0-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f %f %f\n"%(i, matsubara_freq[i], G0[0][i].real, G0[0][i].imag, G0[1][i].real, G0[1][i].imag))
        f.close()

        f=open("Ur-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f\n"%(i, matsubara_freq_boson[i], U_retarded[i].real, U_retarded[i].imag))
        f.close()

    #### hybridization function ####
    hyb=1J*matsubara_freq+vmu-1.0/G0-ek_mean
    for isp in range(2):
        ft_to_tau_hyb(ndiv_tau,vbeta,matsubara_freq,tau,ek_var,hyb[isp],hyb_tau[isp],ndiv_tau/2)
    if PH:
        for isp in range(2):
            for i in range(ndiv_tau):
                rtmp = 0.5*(hyb_tau[isp][i]+hyb_tau[isp][ndiv_tau-i])
                hyb_tau[isp][i] = rtmp
                hyb_tau[isp][ndiv_tau-i] = rtmp

    if mpi.rank==0:
        f=open("delta_n-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f %f %f\n"%(i, matsubara_freq[i], hyb[0][i].real, hyb[0][i].imag, hyb[1][i].real, hyb[1][i].imag))
        f.close()
    
        f=open("delta-raw-"+str(isc)+".dat","w")
        for i in range(ndiv_tau+1):
            f.write("%i %f %f\n"%(i,  hyb_tau[0][i], hyb_tau[1][i]))
        f.close()
   
    #### impurity solver ####
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
    'DELTA'               : "delta-"+str(isc)+".dat",
    'RET_INT_K'           : "K-"+str(isc)+".dat",
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
        f=open("delta-"+str(isc)+".dat","w")
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
        foutput='input_hyb.out.h5'
        f = h5py.File('input_hyb.h5', 'w')
        subgroup = f.create_group('parameters')
        for k,v in imp_parms.iteritems():
            if isinstance(v,int):
                dset = subgroup.create_dataset(k,data=v,dtype='i4') #hybridization does not accept 64-bit integers.
            else:
                dset = subgroup.create_dataset(k, data=v)
        f.close()
        cmd=app_parms['CMD_MPI']+' '+str(app_parms['N_MPI_PROCESS'])+' '+str(app_parms['HYB_PATH'])+' input_hyb.h5 > output-hyb-'+str(isc)
        print cmd
        os.system(cmd)
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
        f=open("self_ene-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f %f\n"%(i, self_ene[0][i].real, self_ene[0][i].imag, self_ene[1][i].real, self_ene[1][i].imag))
        f.close()

        f=open("PI-"+str(isc)+".dat","w")
        for i in range(ndiv_tau):
            f.write("%i %f %f %f\n"%(i, matsubara_freq_boson[i], PI_imp[i].real, PI_imp[i].imag))
        f.close()

    #debug PI should be real and not larger than 0.
    if app_parms.has_key('NON_POSITIVE_PI'):
        if app_parms['NON_POSITIVE_PI']==1:
            for i in range(ndiv_tau):
                PI_imp[i]=min(PI_imp[i].real,-1E-10)

#f=open("G0.dat","w")
#f.write("%i\n"%(ndiv_tau))
#for i in range(ndiv_tau):
    #f.write("%i %f %f %f %f\n"%(i, G0[0][i].real, G0[0][i].imag, G0[1][i].real, G0[1][i].imag))
