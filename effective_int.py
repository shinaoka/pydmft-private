import numpy as np
import sys
import pickle
from fourier_transform import *
from DOS_Cubic_vk import *

def load_retarded_UV(app_parms, INT_TYPE="SCR"):
    ntau =app_parms['NMATSUBARA']
    beta =app_parms['BETA']
    GRID = app_parms['N_K_DIV_IVK_DOS']
    BINS = app_parms['N_GRID_IVK_DOS']-1
    omega_low = app_parms['OMEGA_LOWER_BOUND_IVK_DOS']

    fname = app_parms['SCR_INT_FILE']

    matsubara_freq=np.zeros((ntau,),dtype=float)
    for im in range(ntau):
        matsubara_freq[im]=((2*im+1)*np.pi)/beta

    f = open(fname+'-parms.dat', 'r')
    parms = pickle.load(f)
    f.close()

    target = -1
    if 'target_band' in parms:
        target = parms['target_band']
    elif 'target_bands' in parms:
        target = parms['target_bands'][0]/2
        print "target ", target
    else:
        raise RuntimError("Missing target_band")

    ndim = parms['ndim']
    Vr = np.load(fname+'-Vr.npy')
    V0r = np.load(fname+'-V0r.npy')
    nomega = Vr.shape[0]
    nk = Vr.shape[1]
    omega_max = parms['emax']
    omega_min = parms['emin']
    if omega_min != 0.0:
        print "Error"

    if 'ONLY_ONSITE_U' in app_parms and app_parms['ONLY_ONSITE_U']>0:
        Vr[:,1:nk] = 0.0
        V0r[1:nk] = 0.0

    U_high_omega = V0r[0].real
    Vnn_high_omega = V0r[parms['ndiv_k']**(ndim-1)].real

    ik = 0
    print "Screened Onsite U=", Vr[:,ik].imag

    omega_list = np.linspace(0.0, omega_max, nomega)
    tau_list = np.linspace(0.0, beta, ntau+1)

    #cutoff at low omega
    omega_begin = 0
    for i in range(nomega):
        if omega_list[i] > omega_low:
            omega_begin = i
            break

    #Uscr
    ik=0
    func = np.array(Vr[:,ik].imag)
    func[0] = 0.0
    func[1:nomega] *= 2.0/(np.pi*omega_list[1:nomega])
    dU = np.trapz(func,omega_list)
    Uscr = U_high_omega+dU

    #Z
    ik=0
    func = np.zeros_like(Vr[:,ik].imag)
    #print "Re Vr", Vr[:,ik].real
    #print "Im Vr", Vr[:,ik].imag
    func[omega_begin:nomega] = Vr[omega_begin:nomega,ik].imag/(np.pi*(omega_list[omega_begin:nomega]**2))
    ZB = np.exp(np.trapz(func,omega_list))

    #Vnn
    Vnn = V0r[parms['ndiv_k']**(ndim-1)]

    #Vnnscr
    #print "debug", parms['ndiv_k'], parms['ndiv_k']**(ndim-1), Vr.shape
    func = np.array(Vr[:,parms['ndiv_k']**(ndim-1)].imag)
    func[0] = 0.0
    func[1:nomega] *= 2.0/(np.pi*omega_list[1:nomega])
    dVnn = np.trapz(func,omega_list)
    Vnn_scr = Vnn_high_omega+dVnn

    print "dU", dU
    print "U(omega=0)=", Vr[0,0].real
    print "U(omega=+infty)=", U_high_omega
    print "Uscr=", Uscr
    print "ZB=", ZB

    print "V_nn(omega=0)=", Vr[0,parms['ndiv_k']**(ndim-1)].real
    print "V_nn(omega=+infty)=", Vnn_high_omega
    print "Vnn_scr=", Vnn_scr

    #Ktau
    nomega_cut = nomega
    if 'OMEGA_CUTOFF' in app_parms:
        nomega_cut = app_parms['OMEGA_CUTOFF']
    Ktau = np.zeros((ntau+1,),dtype=float)
    func = np.zeros_like(Vr[:,ik].imag)
    func[omega_begin:nomega_cut] = Vr[omega_begin:nomega_cut,ik].imag/(np.pi*(omega_list[omega_begin:nomega_cut]**2))
    for i in range(ntau+1):
        tau = tau_list[i]
        func2 = np.zeros_like(func)
        func2[omega_begin:nomega_cut] = np.cosh((tau-0.5*beta)*omega_list[omega_begin:nomega_cut])/np.sinh(omega_list[omega_begin:nomega_cut]*0.5*beta)-np.cosh((-0.5*beta)*omega_list[omega_begin:nomega_cut])/np.sinh(omega_list[omega_begin:nomega_cut]*0.5*beta)
        Ktau[i] = np.trapz(func*func2, omega_list)

    #Wtau
    ik=0
    Wtau = np.zeros((ntau+1,),dtype=float)
    func = np.zeros_like(Vr[:,ik].imag)
    func[omega_begin:nomega] = Vr[omega_begin:nomega,ik].imag/np.pi
    #for i in range(ntau+1):
        #tau = tau_list[i]
        #func2 = np.zeros_like(func)
        #func2[omega_begin:nomega] = np.cosh((tau-0.5*beta)*omega_list[omega_begin:nomega])/np.sinh(omega_list[omega_begin:nomega]*0.5*beta)
        #Wtau[i] = np.trapz(func*func2, omega_list)
    #W(i nu_n)
    #W_n = to_freq_bosonic_real_field(ntau,beta,ntau,Wtau,ntau/2)
    W_n = np.zeros((ntau,),dtype=complex)
    for i in range(ntau):
        func3 = np.zeros((len(func),),dtype=complex)
        func3[omega_begin:nomega] = 2*(omega_list[omega_begin:nomega]/(omega_list[omega_begin:nomega]**2+matsubara_freq[i]**2))
        W_n[i] = np.trapz(func*func3, omega_list)

    #W_nn(tau)
    ik=parms['ndiv_k']**(ndim-1)
    func = np.zeros_like(Vr[:,ik].imag)
    func[omega_begin:nomega] = Vr[omega_begin:nomega,ik].imag/np.pi
    Wnn_n = np.zeros((ntau,),dtype=complex)
    for i in range(ntau):
        func3 = np.zeros((len(func),),dtype=complex)
        func3[omega_begin:nomega] = 2*(omega_list[omega_begin:nomega]/(omega_list[omega_begin:nomega]**2+matsubara_freq[i]**2))
        Wnn_n[i] = np.trapz(func*func3, omega_list)

    #print "W_n"
    #for i in range(ntau):
        #print i, W_n[i].real, W_n2[i].real
    #sys.exit(1)

    #lower and upper bounds
    #v_k^{-1}(n) for an effective model with retarded U and Vnn
    ivk_dos = np.zeros((ntau,2,BINS+1),dtype=float)

    ZB_return = 1.0
    if INT_TYPE=='SCR':
        for i in range(ntau):
            lower,upper = get_bounds(GRID, BINS, U_high_omega+W_n[i].real, Vnn_high_omega+Wnn_n[i].real)
            print "iomega, bounds=", i, lower, upper, U_high_omega, W_n[i].real, Vnn_high_omega, Wnn_n[i].real
            if lower==upper:
                if lower>0:
                     lower = upper/1.0001
                     upper *= 1.0001
                else:
                     lower = upper*1.0001
                     upper /= 1.0001
                print "Modified: iomega, bounds=", i, lower, upper
            r = gen_ivk_dos(GRID, BINS, U_high_omega+W_n[i].real, Vnn_high_omega+Wnn_n[i].real, lower, upper)
            ivk_dos[i,:,:] = r[:,:]
    elif INT_TYPE=='UNSCR':
        for i in range(ntau):
            lower,upper = get_bounds(GRID, BINS, U_high_omega, Vnn_high_omega)
            print "iomega, bounds=", i, lower, upper
            if lower==upper:
                lower = upper/1.0001
                upper *= 1.0001
                print "Modified: iomega, bounds=", i, lower, upper
            r = gen_ivk_dos(GRID, BINS, U_high_omega, Vnn_high_omega, lower, upper)
            ivk_dos[i,:,:] = r[:,:]
    elif INT_TYPE=='STATIC':
        ZB_return = ZB
        for i in range(ntau):
            lower,upper = get_bounds(GRID, BINS, Uscr, Vnn_scr)
            print "iomega, bounds=", i, lower, upper
            if lower==upper:
                lower = upper/1.0001
                upper *= 1.0001
                print "Modified: iomega, bounds=", i, lower, upper
            r = gen_ivk_dos(GRID, BINS, Uscr, Vnn_scr, lower, upper)
            ivk_dos[i,:,:] = r[:,:]
    else:
        raise RuntimeError("No corresponding INT_TYPE")

    print "ZB_return", ZB_return
    return ivk_dos,U_high_omega,ZB_return
