import numpy as np
from numpy import cos,pi
from numpy.linalg import inv
import sys
import input_parms as inp
import os
from lib import *
import scipy.optimize as sciopt
from h5dump import *

def solve(app_parms):
    norb=app_parms["norb"]
    nkdiv=app_parms["ndiv_k"]
    Nk=app_parms["nk"]

    #Initialization of H(k)
    H0 = np.zeros((norb,norb),dtype=float)#We assume these are real.
    target = app_parms['target_band']
    for iorb in range(norb):
        for iorb2 in range(norb):
            if iorb!=iorb2 and (iorb==target or iorb2==target):
                H0[iorb,iorb2] = -app_parms["tp"]

    for iorb in range(norb):
        H0[iorb,iorb] = app_parms["e"+str(iorb)]

    Hnn = np.zeros((norb,norb),dtype=float)
    for i in range(norb):
        Hnn[i,i] = -1.0*app_parms["t"+str(i)]

    #H0
    evals,evecs = eigh_ordered(H0)
    print "H0 = ", H0
    print "Evals of H0 = "
    print evals

    Hk = np.zeros((Nk,norb,norb),dtype=float)
    ik = 0
    cos_list = {}
    if app_parms['ndim']==3:
        for i in range(nkdiv):
            for j in range(nkdiv):
                for k in range(nkdiv):
                    Hk[ik,:,:] = H0[:,:]+2.0*(cos(2*i*pi/nkdiv)+cos(2*j*pi/nkdiv)+cos(2*k*pi/nkdiv))*Hnn[:,:]
                    cos_sum = cos(2*i*pi/nkdiv)+cos(2*j*pi/nkdiv)+cos(2*k*pi/nkdiv)
                    if cos_sum in cos_list:
                        cos_list[cos_sum] += 1
                    else:
                        cos_list[cos_sum] = 1
                    ik+=1
    elif app_parms['ndim']==1:
        for ik in range(nkdiv):
            Hk[ik,:,:] = H0[:,:]+2.0*cos(2*i*pi/nkdiv)*Hnn[:,:]
    else:
        raise RuntimeError("ERROR")

    result = {}

    #### self-consistent loop ####
    def calc_diff_HF(n_in):
        #Mean-field Hamiltonian
        evals_list = np.zeros((norb*Nk,),dtype=float)
        evecs_list = []
        for ik in range(Nk):
            H0_mean = np.array(Hk[ik,:,:])
            for iorb in range(norb):
                H0_mean[iorb,iorb] += n_in[iorb]*app_parms['U_'+str(iorb)+'_'+str(iorb)]
                for jorb in range(norb):
                    if iorb!=jorb:
                        U = app_parms['U_'+str(max(iorb,jorb))+'_'+str(min(iorb,jorb))]
                        H0_mean[iorb,iorb] += 2*n_in[jorb]*U
            if ik==0:
                print "dE", np.diag(H0_mean-Hk[ik,:,:])

            #Solve
            evals,evecs = eigh_ordered(H0_mean)
            evals_list[ik*norb:(ik+1)*norb] = 1.0*evals[:]
            evecs_list.append(evecs)
            if ik==0:
                print "eigenvalues", evals

            if ik==0:
                print "evals at ik==0 : ", evals
                result['evecs_k0'] = evecs
                result['evals_k0'] = evals

        n_out = np.zeros_like(n_in)
        idx=np.argsort(evals_list)
        Ne = int(Nk*norb*0.5)
        mu = 0.5*(evals_list[idx[Ne-1]]+evals_list[idx[Ne]])
        for i in range(Ne):
            ie = idx[i]%norb
            ik = idx[i]/norb
            n_out[:] += evecs_list[ik][:,ie]**2
        n_out /= Nk
        F = n_out-n_in
        result["mu"] = mu
        print "n_out ", n_out

        #density matrix
        dmat = np.zeros((norb,norb),dtype=float)
        idx=np.argsort(evals_list)
        Ne = int(Nk*norb*0.5)
        for i in range(Ne):
            ie = idx[i]%norb
            ik = idx[i]/norb
            evec_tmp = np.array(evecs_list[ik][:,ie]).reshape([1,norb])
            dmat += np.dot(evec_tmp.transpose(),evec_tmp)
        dmat /= Nk
        result["dmat"] = dmat

        sys.stdout.flush()
        return F

    n_mean = np.zeros((norb,),dtype=float)
    for iorb in range((norb-1)/2):
        if iorb==(norb-1)/2:
            n_mean[iorb] = 0.5
        else:
            n_mean[iorb] = 1.0

    mix = 0.1
    #n_mean_conv = sciopt.diagbroyden(calc_diff_HF,n_mean,verbose=True)
    n_mean_conv = sciopt.linearmixing(calc_diff_HF,n_mean,verbose=True)
    print n_mean_conv

    dE = np.zeros((norb),dtype=float)
    for iorb in range(norb):
        dE[iorb] += n_mean_conv[iorb]*app_parms['U_'+str(iorb)+'_'+str(iorb)]
        for jorb in range(norb):
            if iorb!=jorb:
                U = app_parms['U_'+str(max(iorb,jorb))+'_'+str(min(iorb,jorb))]
                dE[iorb] += 2*n_mean_conv[jorb]*U

    evals,evecs = eigh_ordered(H0)
    dmat = result["dmat"]
    n_eigen = np.zeros((norb),dtype=float)
    for ie in range(norb):
        n_eigen[ie] = np.dot(evecs[ie], np.dot(dmat,evecs[ie].transpose()))

    return dE, n_eigen, result

def solve_general_lattice(app_parms):
    norb=app_parms["norb"]
    print "norb",norb
    Nk=app_parms["Nk"]
    #H0=app_parms["H0"]
    Hk=app_parms["Hk"]
    fill=app_parms["filling"]
    result = {}

    #### self-consistent loop ####
    def calc_diff_HF(n_in):
        #Mean-field Hamiltonian
        evals_list = np.zeros((norb*Nk,),dtype=float)
        evecs_list = []
        for ik in range(Nk):
            H0_mean = np.array(Hk[ik,:,:])
            for iorb in range(norb):
                H0_mean[iorb,iorb] += n_in[iorb]*app_parms['U_'+str(iorb)+'_'+str(iorb)]
                for jorb in range(norb):
                    if iorb!=jorb:
                        U = app_parms['U_'+str(max(iorb,jorb))+'_'+str(min(iorb,jorb))]
                        H0_mean[iorb,iorb] += 2*n_in[jorb]*U
            if ik==0:
                print "dE", np.diag(H0_mean-Hk[ik,:,:])

            #Solve
            evals,evecs = eigh_ordered(H0_mean)
            evals_list[ik*norb:(ik+1)*norb] = 1.0*evals[:]
            evecs_list.append(evecs)
            if ik==0:
                print "eigenvalues", evals

            if ik==0:
                print "evals at ik==0 : ", evals

        n_out = np.zeros_like(n_in)
        idx=np.argsort(evals_list)
        Ne = int(Nk*norb*0.5)
        mu = 0.5*(evals_list[idx[Ne-1]]+evals_list[idx[Ne]])
        for i in range(Ne):
            ie = idx[i]%norb
            ik = idx[i]/norb
            n_out[:] += evecs_list[ik][:,ie]**2
        n_out /= Nk
        F = n_out-n_in
        print "mu ", mu
        print "n_out ", n_out
        result["mu"] = mu

        #density matrix
        dmat = np.zeros((norb,norb),dtype=float)
        idx=np.argsort(evals_list)
        Ne = int(Nk*fill)
        for i in range(Ne):
            ie = idx[i]%norb
            ik = idx[i]/norb
            evec_tmp = np.array(evecs_list[ik][:,ie]).reshape([1,norb])
            #print "debug", evec_tmp.shape, np.dot(evec_tmp.transpose(),evec_tmp).shape, dmat.shape
            dmat += np.dot(evec_tmp.transpose(),evec_tmp)
        dmat /= Nk
        result["dmat"] = dmat

        sys.stdout.flush()
        return F

    n_mean = np.zeros((norb,),dtype=float)
    for iorb in range((norb-1)/2):
        if iorb==(norb-1)/2:
            n_mean[iorb] = 0.5
        else:
            n_mean[iorb] = 1.0

    mix = 0.1
    n_mean_conv = sciopt.diagbroyden(calc_diff_HF,n_mean,verbose=True)

    dE = np.zeros((norb),dtype=float)
    for iorb in range(norb):
        dE[iorb] += n_mean_conv[iorb]*app_parms['U_'+str(iorb)+'_'+str(iorb)]
        for jorb in range(norb):
            if iorb!=jorb:
                U = app_parms['U_'+str(max(iorb,jorb))+'_'+str(min(iorb,jorb))]
                dE[iorb] += 2*n_mean_conv[jorb]*U

    evals,evecs = eigh_ordered(np.average(Hk,axis=0))
    n_eigen = np.zeros((norb),dtype=float)
    dmat = result["dmat"]
    for ie in range(norb):
        n_eigen[ie] = np.dot(evecs[ie], np.dot(dmat,evecs[ie].transpose()))

    return dE, n_eigen, result
