import numpy as np
import sys
import time
import os
import random
import downfolding_spinless as df
import HF
from lib import *
from math import sin, cos, pi, sqrt

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle

def plot(norb,target,tp,delta,U,V,Vnn,Vnn2,z,ip,fname,plt_show):
    #mu = 0.5*(-sqrt(delta**2+4*tp**2))+0.5*np.sum(np.array(delta))
    print "tp=",tp
    mu = 0.0
    ndim = 3
    print "mu= ", mu
    ndiv_k = 10
    parms={
            'eh_symm' : 1,
            'ndim' : ndim,
            'norb' :  norb,
            'target_band' :  target,
            'z' :  z,
            'e0' : 0,
            'e1' : delta,
            't0'  :  1.0,
            't1'  :  1.0,
            'tp' :  tp,
            'V_0_0_nn' :   Vnn,
            'V_1_0_nn' :   0.,
            'V_1_1_nn' :   Vnn,
            'emax' :   60.0,
            'emin' :   0.,
            'ef' : mu,
            'nk' : ndiv_k**ndim,
            'ndiv_k' : ndiv_k,
            'nomega' : 200,
            'delta' : 0.1,
            }

    for i in range(norb):
        #onsite U
        if i==target:
            parms['U_'+str(i)+'_'+str(i)] = U[0]
        else:
            parms['U_'+str(i)+'_'+str(i)] = U[1]
        #transfer along chain
        parms['t'+str(i)] = 1.0
        #NN V along chain
        parms['V_'+str(i)+'_'+str(i)+'_nn'] = Vnn
        #chemical potential
        parms['e'+str(i)] = delta[i]

    #Hybridization
    for i in range(1,norb):
        for j in range(norb):
            if i==j:
                continue
            parms['U_'+str(i)+'_'+str(j)] = V

    #NN V off diag
    for i in range(norb):
        for j in range(norb):
            if i != j:
                parms['V_'+str(i)+'_'+str(j)+'_nn'] = Vnn2

    norb=parms['norb']
    nomega=parms['nomega']
    nk=parms['nk']

    #HF
    dE_HF = HF.solve(parms)
    for i in range(norb):
        parms['e'+str(i)] += dE_HF[i]

    #Determine chemical potential
    df1d=df.DownFolding2D(parms)
    df1d.Init_band()
    evals_list = np.empty((norb,nk),dtype=float)
    for i in range(nk):
        evals,evec = eigh_ordered(df1d.get_Hk(df1d.kvec[i,:]))
        evals_list[:,i] = 1.0*evals[:]
    evals_list = evals_list.reshape(norb*nk)
    evals_list.sort()
    print evals_list
    ne = nk/2 + nk*parms['target_band']
    mu = 0.5*(evals_list[ne] + evals_list[ne+1])
    parms['ef'] = mu
    print "ef=", mu

    print "Saving parms..."
    fw = open(fname+"-parms.dat","w")
    pickle.dump(parms,fw)
    fw.close()
    
    df1d=df.DownFolding2D(parms)
    print parms
    print "Init_band"
    df1d.Init_band()
    
    df1d.Wannier_fit()

    print "Init_interaction"
    df1d.Init_Interaction()

    print "Cal_"
    time1 = time.clock()
    df1d.Cal_screened_interaction()
    time2 = time.clock()
    print "Cal_screened_interaction took " + str(time2-time1) + "sec."

    print "Proj"
    df1d.Project_Interaction()
    time3 = time.clock()
    print "Cal_screened_interaction took " + str(time3-time2) + "sec."
    
    orb=0
    omega=0
    
    print "Eigen_enes at Gamma = ", df1d.ek[0,:]
    print "Eigen_enes at Gamma = ", df1d.ek[1,:]
    print "Eigen_enes at Gamma = ", df1d.ek[2,:]
    print "Eigen_vecs at Gamma = ", df1d.wf[0,0,:]
    print "Eigen_vecs at Gamma = ", df1d.wf[0,1,:]
    print "Eigen_vecs at Gamma = ", df1d.wf[0,2,:]
    
    plt.figure(1,figsize=(8,12))
    plt.subplots_adjust(hspace=0.4)
    
    #Band structure
    plt.subplot(411)
    title_str = r"$N_\mathrm{orb}$="+str(norb)+r", $t_\perp$="+str(tp)+r", $U_d$="+str(U[0])+r", $U_r$="+str(U[1])+r", $U_{dr}$="+str(V)+r", $V_\mathrm{nn}^=$="+str(Vnn)+r", $V_\mathrm{nn}^\times$="+str(Vnn2)
    plt.title(title_str, fontname='serif')

    plt.xlabel(r'$k$', fontname='serif')
    plt.ylabel(r'$E-E_\mathrm{F}$', fontname='serif')
    plt.xlim([0,3.0])
    plt.xticks( (0,1.0,2.0,3.0), ("0", r"$(\pi,0,0)$", r"$(\pi,\pi,0)$", r"$(\pi,\pi,\pi)$"))
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')
    M=ndiv_k/2+1
    xdata = np.zeros((3*M,),dtype=float)
    ydata = np.zeros((3*M,norb),dtype=float)
    for ik in range(M):
        kx = pi*float(ik)/(M-1)
        xdata[ik] = float(ik)/(M-1)
        evals,evec = eigh_ordered(df1d.get_Hk([kx,0.0,0.0]))
        ydata[ik,:] = evals-mu
    for ik in range(M):
        kx = pi
        ky = pi*float(ik)/(M-1)
        xdata[ik+M] = 1.0+float(ik)/(M-1)
        evals,evec = eigh_ordered(df1d.get_Hk([kx,ky,0.0]))
        ydata[ik+M,:] = evals-mu
    for ik in range(M):
        kx = pi
        ky = pi
        kz = pi*float(ik)/(M-1)
        xdata[ik+2*M] = 2.0+float(ik)/(M-1)
        evals,evec = eigh_ordered(df1d.get_Hk([kx,ky,kz]))
        ydata[ik+2*M,:] = evals-mu

    for i in range(norb):
        plt.plot(xdata, ydata, label=r"", marker='x', linestyle='-', markersize=5)
    xdata = [0,3]
    ydata = [0.0,0.0]
    plt.plot(xdata, ydata, label=r"", marker='', linestyle='-', color="black", markersize=5)

    #Density of states
    plt.subplot(412)
    plt.xlabel(r'$E-E_\mathrm{F}$', fontname='serif')
    plt.ylabel(r'Density of states', fontname='serif')
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')
    M=1000
    evals_list = np.empty((norb,M),dtype=float)
    for i in range(M):
        kvec = [2.0*pi*random.random() for j in range(ndim)]
        evals,evec = eigh_ordered(df1d.get_Hk(kvec))
        evals_list[:,i] = 1.0*evals[:]-mu
    for i in range(norb):
        plt.hist(evals_list[i,:],25)

    
    print "U (unscreened/screened) projected on the target band"
    ik=0
    print "static=", df1d.V0r[ik,target,target].real
    print "Screened=", df1d.Vr[:,ik,target,target].real

    print "Vnn (unscreened/screened) projected on the target band"
    ik=parms['ndiv_k']**(ndim-1)
    print "static=", df1d.V0r[ik,target,target].real
    print "Screened=", df1d.Vr[:,ik,target,target].real

    np.save(fname+"-Vr", df1d.Vr[:,:,target,target])
    np.save(fname+"-V0r", df1d.V0r[:,target,target])

    #ONSITE
    plt.subplot(413)
    plt.xlabel(r'$\omega$', fontname='serif')
    plt.ylabel(r'Onsite Interaction', fontname='serif')
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')
    
    #for i in range(norb):
        #print "unscreened U ", i, df1d.W0r[0,i,i].real, df1d.V0r[0,i,i].real

    ydata = df1d.Vr[:,0,target,target]
    plt.plot(df1d.omega, ydata, label=r"U", marker='o', linestyle='-', markersize=5)
    ydata = [df1d.V0r[0,target,target]]*len(df1d.omega)
    plt.plot(df1d.omega, ydata, label=r"U (unscreened)", marker='', linestyle='-', color='blue', markersize=5)
    ax = plt.twinx()
    ydata = df1d.Vr[:,0,target,target].imag
    plt.plot(df1d.omega, ydata, label=r"U", marker='x', linestyle='--', color='gray', markersize=5)
    plt.plot(df1d.omega, [0.0]*len(ydata), label=r"U", marker='', linestyle='--', color='gray', markersize=5)

    #NN
    plt.subplot(414)
    plt.xlabel(r'$\omega$', fontname='serif')
    plt.ylabel(r'NN Interaction', fontname='serif')
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')

    ydata = df1d.Vr[:,parms['ndiv_k']**(ndim-1),target,target]
    plt.plot(df1d.omega, ydata, label=r"NN V", marker='o', linestyle='-', color='blue', markersize=5)
    ydata = [df1d.V0r[parms['ndiv_k']**(ndim-1),target,target]]*len(df1d.omega)
    plt.plot(df1d.omega, ydata, label=r"NN V (unscreened)", marker='', linestyle='-', color='blue', markersize=5)

    ax = plt.twinx()
    ydata = df1d.Vr[:,parms['ndiv_k']**(ndim-1),target,target].imag
    plt.plot(df1d.omega, ydata, label=r"U", marker='x', linestyle='--', color='gray', markersize=5)
    plt.plot(df1d.omega, [0.0]*len(ydata), label=r"U", marker='', linestyle='--', color='gray', markersize=5)

    plt.savefig(fname+".pdf")
    if plt_show:
        plt.show()
    plt.close()

plt_show = False
param_set = []
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 0.0, 0.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 5.0, 0.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 5.0, 5.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 5.0, 5.0, 5.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 5.0, 0.0, 5.0, 1.0])
#param_set.append([3, 1, 2.0, [-10.0,0.0,10.0], [10.0, 5.0], 0.0, 0.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-10.0,0.0,10.0], [10.0, 5.0], 5.0, 0.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-10.0,0.0,10.0], [10.0, 5.0], 5.0, 5.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-10.0,0.0,10.0], [10.0, 5.0], 5.0, 5.0, 5.0, 1.0])
#param_set.append([3, 1, 2.0, [-10.0,0.0,10.0], [10.0, 5.0], 5.0, 0.0, 5.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [20.0, 5.0], 5.0, 5.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 0.0, 5.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 0.0], 0.0, 5.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 0.0], 0.0, 1.0, 0.0, 1.0])
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 5.0, 0.0, 0.0, 1.0]) #Fig15
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 5.0, 1.0, 0.0, 1.0]) #Fig16
#param_set.append([3, 1, 2.0, [-10.0,0.0,10.0], [10.0, 5.0], 5.0, 5.0, 0.0, 1.0]) #Fig17
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [15.0, 7.5], 7.5, 0.0, 0.0, 1.0]) #Fig18
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [0.0, 0.0], 0.0, 0.0, 0.0, 1.0]) #Fig19
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [20.0, 10.0], 10.0, 0.0, 0.0, 1.0]) #Fig20
#param_set.append([3, 1, 4.0, [-7.0,0.0,7.0], [20.0, 10.0], 10.0, 0.0, 0.0, 1.0]) #Fig21
#param_set.append([5, 2, 2.0, [-8.0,-8.0,0.0,8.0,8.0], [20.0, 10.0], 10.0, 0.0, 0.0, 1.0]) #Fig22
#param_set.append([5, 2, 2.0, [-8.0,-8.0,0.0,8.0,8.0], [20.0, 10.0], 0.0, 0.0, 0.0, 1.0]) #Fig23
#
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [10.0, 5.0], 0.0, 0.0, 0.0, 1.0]) #Fig24
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [20.0, 10], 0.0, 0.0, 0.0, 1.0]) #Fig25
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [20.0, 20], 0.0, 0.0, 0.0, 1.0]) #Fig26
#
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [15.0, 7.5], 0.0, 0.0, 0.0, 1.0]) #Fig27
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [17.5, 8.75], 0.0, 0.0, 0.0, 1.0]) #Fig28

Ud=10
param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.25*Ud, 0.0, 0.0, 1.0, "Up0.25-Ud10"])

Ud=12.5
param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.25*Ud, 0.0, 0.0, 1.0, "Up0.25-Ud12.5"])

Ud=15.0
param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.25*Ud, 0.0, 0.0, 1.0, "Up0.25-Ud15"])

Ud=17.5
param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.25*Ud, 0.0, 0.0, 1.0, "Up0.25-Ud17.5"])

Ud=20.0
param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.25*Ud, 0.0, 0.0, 1.0, "Up0.25-Ud20.0"])


#Up=0.5
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.5*Ud, 0.0, 0.0, 1.0]) #Fig34
#
#Ud=10.0
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.5*Ud, 0.0, 0.0, 1.0, "Up0.5-Ud10"]) #35
#Ud=12.0
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.5*Ud, 0.0, 0.0, 1.0, "Up0.5-Ud12"]) #36
#Ud=15.0
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.5*Ud, 0.0, 0.0, 1.0, "Up0.5-Ud15"]) #37
#Ud=20.0
#param_set.append([3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], 0.5*Ud, 0.0, 0.0, 1.0, "Up0.5-Ud15"]) #38


#istart=28
#iend=33
#for i in range(len(param_set)):
    #ip=i+1
    #param = param_set[i]
    #plot(param[0],param[1],param[2],param[3],param[4],param[5],param[6],param[7],param[8],ip,param[9],plt_show)

#Ud_list = np.linspace(10.0,30.0,21)
#Ud_list = np.linspace(14.0,16.0,9)
#Ud_list = np.array([0.0,5.0])
norb=3
Up_list = np.array([0.5])
for Up in Up_list:
    Ud_list = np.linspace(10.0,10.0,1)
    print Up,Ud_list
    for Ud in Ud_list:
        print Up,Ud
        if norb==3:
            param = [3, 1, 2.0, [-7.0,0.0,7.0], [Ud, 0.5*Ud], Up*Ud, 0.0, 0.0, 1.0, "Up"+str(Up)+"-Ud"+str(Ud)]
            plot(param[0],param[1],param[2],param[3],param[4],param[5],param[6],param[7],param[8],1,param[9],plt_show)
        elif norb==5:
            gap = 15.0
            param = [5, 2, 2.0, [-gap,-gap,0.0,gap,gap], [Ud, 0.5*Ud], Up*Ud, 0.0, 0.0, 1.0, "5orb-large-gap-Up"+str(Up)+"-Ud"+str(Ud)]
            plot(param[0],param[1],param[2],param[3],param[4],param[5],param[6],param[7],param[8],1,param[9],plt_show)
        elif norb==7:
            gap = 7.0
            param = [7, 3, 2.0, [-gap,-gap,-gap,0.0,gap,gap,gap], [Ud, 0.5*Ud], Up*Ud, 0.0, 0.0, 1.0, "7orb-Up"+str(Up)+"-Ud"+str(Ud)]
            plot(param[0],param[1],param[2],param[3],param[4],param[5],param[6],param[7],param[8],1,param[9],plt_show)
