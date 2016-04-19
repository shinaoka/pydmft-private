import numpy as np
from numpy.linalg import inv
import sys, copy, random
import time
import integration as intgr
import re 
import input_parms as inp
import os
import subprocess,shlex
import h5py
import fourier_transform as ft
import tempfile
from collections import OrderedDict
from lib import *
from h5dump import *

def load_sign(hf):
    return hf['/simulation/results/Sign/mean/value'].value

def load_obs_with_sign(hf,obs):
    sign = hf['/simulation/results/Sign/mean/value'].value
    return (hf['/simulation/results/'+obs+'_Re/mean/value'].value+1J*hf['/simulation/results/'+obs+'_Im/mean/value'].value)/sign

def load_real_obs_with_sign(hf,obs):
    sign = hf['/simulation/results/Sign/mean/value'].value
    return hf['/simulation/results/'+obs+'/mean/value'].value/sign

def write_Utensor_cthyb_alpscore(fname, tensor):
    f = open(fname,'w')
    N1 = tensor.shape[0]
    N2 = tensor.shape[1]
    N3 = tensor.shape[2]
    N4 = tensor.shape[3]
    num_elem = 0
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                for i4 in range(N4):
                    if np.abs(tensor[i1,i2,i3,i4]) > 0.0:
                        num_elem += 1

    print>>f, num_elem
    i_elem = 0
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                for i4 in range(N4):
                    if np.abs(tensor[i1,i2,i3,i4]) > 0.0:
                        print >>f, i_elem, i1, i2, i3, i4, tensor[i1,i2,i3,i4].real, tensor[i1,i2,i3,i4].imag
                        i_elem += 1
    f.close()

#Return U**alpha
def unitary_mat_power(Umat, alpha):
    assert Umat.shape[0]==Umat.shape[1]
    N = Umat.shape[0]
    evals,Vmat = np.linalg.eig(Umat)
    Gamma = np.zeros((N,N),dtype=complex)
    for i in xrange(N):
        Gamma[i,i] = evals[i]**alpha
    return np.dot(np.dot(Vmat,Gamma),Vmat.conjugate().transpose())

def integrate_hyb(hyb_tau):
    ntau = hyb_tau.shape[0]-1 
    nf = hyb_tau.shape[1]

    hyb = np.zeros((nf,nf),dtype=complex)
    for itau in xrange(ntau+1):
        hyb[:,:] += np.dot(hyb_tau[itau,:,:].conjugate().transpose(),hyb_tau[itau,:,:])

    return hyb

def diagonalize_hyb(hyb_tau, Umat):
    ntau = hyb_tau.shape[0]-1 
    nf_sbl = Umat.shape[0]

    hyb_tau_prj = projection(hyb_tau[:,0:nf_sbl,0:nf_sbl], Umat, nf_sbl)
    for itau in xrange(ntau+1):
        for i in xrange(nf_sbl):
            for j in xrange(i):
                hyb_tau_prj[itau,i,j] = 0.0
                hyb_tau_prj[itau,j,i] = 0.0
            
    return projection(hyb_tau_prj, Umat.conjugate().transpose(), nf_sbl)


def symmetrize_G_tau(app_parms, G_tau):
    ntau = G_tau.shape[0]-1 
    nflavor_sbl = G_tau.shape[1]
    assert G_tau.shape[1]==G_tau.shape[2]

    G_tau_new = np.zeros_like(G_tau)
    if 'SYMM_MAT' in app_parms:
        nsymm = app_parms['SYMM_MAT'].shape[0]
        assert app_parms['SYMM_MAT'].shape[1]==nflavor_sbl
        assert app_parms['SYMM_MAT'].shape[2]==nflavor_sbl
        print "Symmetrizing G_tau..."

        G_tau_symm = np.zeros((nsymm+1,ntau+1,nflavor_sbl,nflavor_sbl),dtype=complex)

        G_tau_symm[0,:,:,:] = 1.0*G_tau
        for isymm in xrange(nsymm):
            G_tau_symm[isymm+1,:,:,:] = projection(G_tau[:,:,:], app_parms['SYMM_MAT'][isymm,:,:],nflavor_sbl)
        G_tau_new[:,:,:] = np.average(G_tau_symm, axis=0)
    else:
        G_tau_new[:,:,:] = 1.*G_tau

    if 'PM' in app_parms and app_parms['PM'] != 0:
            print "Making G_tau paramagnetic..."
            for iorb in range(nflavor_sbl/2):
                #mz=0
                G_tau_new[:,2*iorb,2*iorb] = 0.5*(G_tau_new[:,2*iorb,2*iorb]+G_tau_new[:,2*iorb+1,2*iorb+1])
                G_tau_new[:,2*iorb+1,2*iorb+1] = 1.0*G_tau_new[:,2*iorb,2*iorb]
                #mx=0 and my=0
                G_tau_new[:,2*iorb,2*iorb+1] = 0.0
                G_tau_new[:,2*iorb+1,2*iorb] = 0.0
    return G_tau_new

#hyb_tau: Delta(\tau), 
# Note: when we convert Delta to F, we have to exchange flavor indices in Delta and rotmat.
def solve_sbl_imp_model(app_parms, imp_model, fourie_transformer, tau_mesh, hyb_tau, hyb, invG0, mu, isbl):
    time1 = time.time()

    ntau = len(tau_mesh)-1
    norb = imp_model.get_norb()
    nsbl = imp_model.get_nsbl()
    nflavor = imp_model.get_nflavor()
    nflavor_sbl = nflavor/nsbl
    norb_sbl = norb/nsbl
    beta = app_parms['BETA']

    start = isbl*nflavor_sbl
    end = (isbl+1)*nflavor_sbl

    #### impurity solver ####
    path_input = app_parms['PREFIX_IMP_SLV_WORK_FILE']+'_input_hyb_sbl'+str(isbl)
    path_hyb = app_parms['PREFIX_IMP_SLV_WORK_FILE']+'_F_sbl'+str(isbl)

    #Generate input files...
    parms=OrderedDict()
    if 'ASSUME_REAL' in app_parms and app_parms['ASSUME_REAL'] != 0:
        parms['ALGORITHM'] = "real-matrix"
    else:
        parms['ALGORITHM'] = "complex-matrix"
    parms['N_TAU'] = app_parms['NMATSUBARA']
    parms['BETA'] = app_parms['BETA']
    parms['SITES'] = norb_sbl
    parms['SPINS'] = 2
    parms['FLAVORS'] = nflavor_sbl
    parms['F_INPUT_FILE'] = path_hyb
    parms['BASIS_INPUT_FILE'] = path_hyb+'-rot_sbl'+str(isbl)

    #Basis rotation for \Delta (not F)
    if app_parms['BASIS_ROT']==0:
        rotmat_Delta = np.identity(nflavor_sbl,dtype=complex)
    elif 'ROTMAT_DELTA' in app_parms:
        rotmat_Delta = app_parms['ROTMAT_DELTA']
    elif float(app_parms['BASIS_ROT'])>0.0: #Diagonalizing sublattice H0
        if (not 'BASIS_ROT_TYPE' in app_parms) or ('BASIS_ROT_TYPE' in app_parms and app_parms['BASIS_ROT_TYPE']==0):
            print "Diagonalizing local Hamiltonian..."
            h_mat = imp_model.get_moment(1)[start:end,start:end]
        else:
            print "Diagonalizing integrated Delta..."
            h_mat = integrate_hyb(hyb_tau[:,start:end,start:end])

	if 'PREVENT_MIXING_SPIN_SECTORS_IN_BASIS_ROT' in app_parms and app_parms['PREVENT_MIXING_SPIN_SECTORS_IN_BASIS_ROT'] != 0:
            print "Using PREVENT_MIXING_SPIN_SECTORS_IN_BASIS_ROT..."
            Hsbl = h_mat.reshape([nflavor_sbl/2,2,nflavor_sbl/2,2])
            evals_up,evecs_up = eigh_ordered(Hsbl[:,0,:,0]) #Diagonal H0_{up,up}
            rotmat_Delta = np.zeros((nflavor_sbl/2,2,nflavor_sbl/2,2),dtype=complex)
            rotmat_Delta[:,0,:,0] = 1.*evecs_up
            rotmat_Delta[:,1,:,1] = 1.*evecs_up
            rotmat_Delta = rotmat_Delta.reshape([nflavor_sbl,nflavor_sbl])
        else:
            evals,evecs = eigh_ordered(h_mat)
            rotmat_Delta = 1.*evecs
        print "Using alpha=", float(app_parms['BASIS_ROT'])
        rotmat_Delta = unitary_mat_power(rotmat_Delta, float(app_parms['BASIS_ROT'])) 

    #Write hyb func
    if 'DIAG_HYB' in app_parms and app_parms['DIAG_HYB'] != 0:
        hyb_tau_sbl = diagonalize_hyb(hyb_tau[:,start:end,start:end], rotmat_Delta)
    else:
        hyb_tau_sbl = 1.*hyb_tau[:,start:end,start:end]

    hyb_f = open(path_hyb,'w')
    for i in range(ntau+1):
        for iflavor in range(nflavor_sbl):
            for iflavor2 in range(nflavor_sbl):
                print >>hyb_f, i, iflavor, iflavor2, -hyb_tau_sbl[ntau-i,iflavor2,iflavor].real, -hyb_tau_sbl[ntau-i,iflavor2,iflavor].imag
    hyb_f.close()

    #Local H0
    parms['HOPPING_MATRIX_INPUT_FILE'] = path_input+'-hopping_matrix.txt'
    write_matrix(path_input+'-hopping_matrix.txt', imp_model.get_H0()[start:end,start:end]-mu*np.identity(nflavor_sbl))

    #U tensor
    parms['U_TENSOR_INPUT_FILE'] = path_input+'-Uijkl.txt'
    write_Utensor_cthyb_alpscore(path_input+'-Uijkl.txt', imp_model.get_Uijkl())

    #Single-particle basis rotation for Delta
    write_matrix(path_hyb+'-rot_sbl'+str(isbl), rotmat_Delta)

    for k,v in app_parms.items():
        m = re.search('^IMP_SLV_(.+)$',k)
        if m!=None:
            print k,v,m.group(0),m.group(1)
            parms[m.group(1)] = v

    #Set random seed
    random.seed()
    parms['SEED'] = random.randint(0,10000)

    #Write parameters
    input_f = open(path_input+'.ini','w')
    write_parms_to_ini(input_f, parms)
    input_f.close()

    if (os.path.exists(path_input+'.out.h5')):
      os.remove(path_input+'.out.h5')

    output_f = open('output_'+path_input, 'w')
    cmd=app_parms['CMD_MPI']+' '+str(app_parms['N_MPI_PROCESS'])+' '+str(app_parms['HYB_PATH'])+' '+path_input+'.ini'
    print cmd
    time2 = time.time()
    args = shlex.split(cmd)
    subprocess.call(args, stdout=output_f, stderr=output_f) # Success!
    output_f.close()
    print "Finished hybridization"
    time3 = time.time()

    #Load measured observables
    result = {}
    foutput=path_input+'.out.h5'

    print "Opening ", foutput, "..."
    hf = h5py.File('./'+foutput, 'r')

    #<Sign>
    sign = load_sign(hf)
    print "sign=", complex(sign)
    print "abs(sign)=", np.abs(sign)
    
    #<n_i> in the rotated basis
    result["n_rotated"] = load_real_obs_with_sign(hf,"n").real
    
    #Im G(tau)
    G_tau = -load_obs_with_sign(hf,"Greens").reshape(2*norb_sbl,2*norb_sbl,ntau+1).transpose([2,0,1])
    G_tau[0,:,:] *= 2 #because the bin size is half at \tau=0 and beta.
    G_tau[ntau,:,:] *= 2
    
    hf.close()

    G_tau_prj = projection(G_tau,rotmat_Delta,2*norb_sbl)
    for iflavor in range(nflavor_sbl):
        G_tau_prj[0,iflavor,iflavor] = -(1.0-result["n_rotated"][iflavor])
        G_tau_prj[ntau,iflavor,iflavor] = -1.0*result["n_rotated"][iflavor]
    G_tau = projection(G_tau_prj,rotmat_Delta.transpose().conjugate(),2*norb_sbl)

    G_tau = symmetrize_G_tau(app_parms, G_tau)
    result["Greens_imag_tau"] = G_tau

    #Load all observables
    keys,means,errors = load_observables("./"+foutput)
    obs = {}
    for i in range(len(keys)):
        obs[keys[i]+'_mean'] = means[i]
        obs[keys[i]+'_error'] = errors[i]

    #Fourie tranformation of G(tau) to G(i\omega_n)
    G_imp = fourie_transformer.G_tau_to_freq2(ntau, beta, G_tau, app_parms["CUTOFF_FOURIE"])
    result["G_imp"] = G_imp

    #Compute self energy
    self_ene_sbl = np.zeros((ntau,nflavor_sbl,nflavor_sbl),dtype=complex)
    for im in range(ntau):
        self_ene_sbl[im,:,:]=invG0[im,:,:]-inv(G_imp[im,:,:])
    result["self_ene"] = self_ene_sbl

    time4 = time.time()

    print "Timings of solving an impurity model tot=", time4-time1, " : ", time2-time1, " ", time3-time2, " ", time4-time3, "isbl=",isbl

    return result, obs

#hyb_tau: Delta(\tau), 
# Note: when we convert Delta to F, we have to exchange flavor indices in Delta and rotmat.
def call_hyb_matrix(app_parms, imp_model, fourie_transformer, tau_mesh, hyb_tau, hyb, invG0, mu):
    ntau = len(tau_mesh)-1
    norb = imp_model.get_norb()
    nsbl = imp_model.get_nsbl()
    nflavor = imp_model.get_nflavor()
    nflavor_sbl = nflavor/nsbl
    norb_sbl = norb/nsbl
    beta = app_parms['BETA']

    single_imp = (not ('MULTI_IMP' in app_parms and app_parms['MULTI_IMP'] != 0))

    if single_imp:
        result,obs = solve_sbl_imp_model(app_parms, imp_model, fourie_transformer, tau_mesh, hyb_tau, hyb, invG0[0,:,:,:], mu, 0)
        #Copy sublattice self-energy to unit-cell self-energy 
        self_ene = np.zeros((ntau,nflavor,nflavor),dtype=complex)
        for isbl in range(nsbl):
            self_ene[:,isbl*nflavor_sbl:(isbl+1)*nflavor_sbl, isbl*nflavor_sbl:(isbl+1)*nflavor_sbl] = 1.*result['self_ene'][:,:,:]
        result["self_ene"] = self_ene
        return result, obs
    else:
        #### solving an impurity problem for each site ####
        results_sbl = []
        obs_sbl = []
        for isbl in xrange(nsbl):
            r,o = solve_sbl_imp_model(app_parms, imp_model, fourie_transformer, tau_mesh, hyb_tau, hyb, invG0[isbl,:,:,:], mu, isbl)
            results_sbl.append(r)
            obs_sbl.append(o)

        result = {}
        obs = {}
    
        #Compute G(tau) and self-energy
        #result["n"] = np.zeros((nflavor,),dtype=float)
        result["n_rotated"] = np.zeros((nflavor,),dtype=float)
        result["Greens_imag_tau"] = np.zeros((ntau+1,nflavor,nflavor),dtype=complex)
        result["G_imp"] = np.zeros((ntau,nflavor,nflavor),dtype=complex)
        result["self_ene"] = np.zeros((ntau,nflavor,nflavor),dtype=complex)
        for isbl in range(nsbl):
            start = isbl*nflavor_sbl
            end = (isbl+1)*nflavor_sbl
    
            #result['n'][start:end] = results_sbl[isbl]['n'][:]
            result['n_rotated'][start:end] = results_sbl[isbl]['n_rotated'][:]
            result["Greens_imag_tau"][:,start:end,start:end] = results_sbl[isbl]['Greens_imag_tau'][:,:,:]
            result["G_imp"][:,start:end,start:end] = results_sbl[isbl]['G_imp'][:,:,:]
            result["self_ene"][:,start:end,start:end] = results_sbl[isbl]['self_ene'][:,:,:]

        #Merge all other data
        for isbl in range(nsbl):
            for k,v in obs_sbl[isbl].items():
                obs[k+"_sbl"+str(isbl)] = v
    
        return result, obs
