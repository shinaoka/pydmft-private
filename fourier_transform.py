import numpy as np
from scipy import interpolate
import scipy
import sys
from lib import *
from scipy.optimize import curve_fit

def mpow(mat, n):
    assert n>=1
    if n==1:
        return mat
    else:
        return np.dot(mat,mpow(mat,n-1))

def ft_to_tau_hyb(ndiv_tau, beta, matsubara_freq, tau, c1, c2, c3, data_n, data_tau,cutoff):
 nflavor = c1.shape[0]
 #tail correction
 #Htmp = - np.dot(H1,np.dot(H1,H1))+2*H1*H2-H3
 tail = np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
 for im in range(ndiv_tau):
     tail[im,:,:] = c1/(1J*matsubara_freq[im])+c2/((1J*matsubara_freq[im])**2)+c3/((1J*matsubara_freq[im])**3)
 data_rest = data_n - tail
 data_rest[cutoff:,:,:] = 0.0
 assert is_hermitian(c1)
 assert is_hermitian(c2)
 assert is_hermitian(c3)

 y = np.zeros((2*ndiv_tau,nflavor,nflavor),dtype=complex)
 for im in range(ndiv_tau):
     y[2*im+1,:,:] = data_rest[im,:,:]
 y *= 1.0/beta

 data_tau[:,:,:] = np.fft.fft(y,axis=0)[0:ndiv_tau+1,:,:]
 data_tau_conj = data_tau.transpose((0,2,1)).conj()
 for it in range(ndiv_tau+1):
     tau_tmp=tau[it]
     data_tau[it,:,:] += data_tau_conj[it,:,:] -0.5*c1 +0.25*c2*(-beta+2*tau_tmp) +0.25*c3*(beta-tau_tmp)*tau_tmp
 return tail, data_rest


#estimate tail
def estimate_tail(ndiv_tau, beta, matsubara_freq, tau, c1_0, data_n):
  cutoff = int(3*ndiv_tau/4)
  nflavor = c1_0.shape[0]

  tail = np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
  for im in range(ndiv_tau):
    tail[im,:,:] = c1_0/(1J*matsubara_freq[im])
  xdata = matsubara_freq[cutoff:ndiv_tau]
  ydata = (data_n-tail)[cutoff:ndiv_tau,:,:]

  def tail_func_real(x, c2_real, c3_imag):
    return - c2_real/x**2 - c3_imag/x**3

  def tail_func_imag(x, c2_imag, c3_real):
    return - c2_imag/x**2 + c3_real/x**3

  #def tail_func_real(x, c1_imag, c2_real, c3_imag):
    #return  c1_imag/x - c2_real/x**2 - c3_imag/x**3

  #def tail_func_imag(x, c1_real, c2_imag, c3_real):
    #return -c1_real/x - c2_imag/x**2 + c3_real/x**3

  #def tail_func(x, a, b, c):
     #return a/(1J*x)+b/((1J*x)**2)+c/((1J*x)**3)

  #c1 = np.zeros((nflavor,nflavor),dtype=complex)
  c1 = 1.*c1_0
  c2 = np.zeros((nflavor,nflavor),dtype=complex)
  c3 = np.zeros((nflavor,nflavor),dtype=complex)
  for flavor in xrange(nflavor):
    for flavor2 in xrange(nflavor):

      p0 = [0.0, 0.0]
      popt,pcov = curve_fit(tail_func_real, xdata, ydata[:,flavor,flavor2].real, p0)
      c2[flavor,flavor2] += popt[0]
      c3[flavor,flavor2] += 1J*popt[1]

      p0 = [0.0, 0.0]
      popt,pcov = curve_fit(tail_func_imag, xdata, ydata[:,flavor,flavor2].imag, p0)
      c2[flavor,flavor2] += 1J*popt[0]
      c3[flavor,flavor2] += popt[1]

  return c1,c2,c3

#Estimate c2, c2 by a fit
def ft_to_tau_hyb_with_fitted_tail(ndiv_tau, beta, matsubara_freq, tau, c1_0, data_n, data_tau):
 nflavor = c1_0.shape[0]

 c1, c2, c3 = estimate_tail(ndiv_tau, beta, matsubara_freq, tau, c1_0, data_n)

 #tail correction
 tail = np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
 for im in range(ndiv_tau):
     tail[im,:,:] = c1/(1J*matsubara_freq[im])+c2/((1J*matsubara_freq[im])**2)+c3/((1J*matsubara_freq[im])**3)
 data_rest = data_n - tail
 assert is_hermitian(c1)

 y = np.zeros((2*ndiv_tau,nflavor,nflavor),dtype=complex)
 for im in range(ndiv_tau):
     y[2*im+1,:,:] = data_rest[im,:,:]
 y *= 1.0/beta

 data_tau[:,:,:] = np.fft.fft(y,axis=0)[0:ndiv_tau+1,:,:]
 data_tau_conj = data_tau.transpose((0,2,1)).conj()
 for it in range(ndiv_tau+1):
     tau_tmp=tau[it]
     data_tau[it,:,:] += data_tau_conj[it,:,:] -0.5*c1 +0.25*c2*(-beta+2*tau_tmp) +0.25*c3*(beta-tau_tmp)*tau_tmp
 return tail, data_rest

def G_tau_to_freq(ntau,beta,G_tau):
    assert len(G_tau)==ntau+1

    f_freq = to_freq_fermionic_real_field(ntau,beta,ntau,G_tau.real,ntau)+1J*to_freq_fermionic_real_field(ntau,beta,ntau,G_tau.imag,ntau)
    assert len(f_freq)==ntau
    return f_freq

def G_tau_to_freq2(ntau,beta,G_tau,cutoff):
    assert G_tau.shape[0]==ntau+1
    assert G_tau.shape[1]== G_tau.shape[2]
    nflavor = G_tau.shape[1]

    G_tau_ex = np.zeros((2*ntau,nflavor,nflavor),dtype=complex)
    G_tau_ex[0:ntau+1,:,:] = G_tau[:,:,:]
    G_tau_ex[(ntau+1):,:,:] = -G_tau[1:ntau,:,:]

    #1/iomega correction
    for itau in xrange(ntau+1):
        for flavor in xrange(nflavor):
            G_tau_ex[itau,flavor,flavor] += 0.5

    for itau in xrange(ntau+1,2*ntau):
        for flavor in xrange(nflavor):
            G_tau_ex[itau,flavor,flavor] += -0.5

    y = np.fft.ifft(G_tau_ex,axis=0)

    #for itau in xrange(2*ntau):
        #print itau, G_tau_ex[itau,0,0].real

    G_omega = np.zeros((ntau,nflavor,nflavor),dtype=complex)
    for im in xrange(cutoff):
        G_omega[im,:,:] = beta*y[2*im+1,:,:]

    omega_n = np.array([(2*n+1)*np.pi/beta for n in xrange(ntau)])
    for im in xrange(ntau):
        for flavor in xrange(nflavor):
           G_omega[im,flavor,flavor] += 1/(1J*omega_n[im])

    return G_omega

class FourieTransformer:
    def __init__(self, model):
        self.nflavor_ = model.get_nflavor()
        self.nsbl_ = model.get_nsbl()
        self.nflavor_sbl_ = self.nflavor_/self.nsbl_
        self.M1_ = hermitialize(model.get_moment(1))
        self.M2_ = hermitialize(model.get_moment(2))
        self.M3_ = hermitialize(model.get_moment(3))
        self.M4_ = hermitialize(model.get_moment(4))

        assert is_hermitian(self.M1_)
        assert is_hermitian(self.M2_)
        assert is_hermitian(self.M3_)
        assert is_hermitian(self.M4_)

        #Compute coefficients for fourie transforming Delta (sign comes from i^n is not included)
        self.c1_ = np.zeros((self.nsbl_,self.nflavor_sbl_,self.nflavor_sbl_),dtype=complex)
        for isbl in xrange(self.nsbl_):
            start = self.nflavor_sbl_*isbl
            end = self.nflavor_sbl_*(isbl+1)
            self.c1_[isbl,:,:] = self.M2_[start:end,start:end]-mpow(self.M1_[start:end,start:end],2)
        self.c2_ = np.zeros((self.nsbl_,self.nflavor_sbl_,self.nflavor_sbl_),dtype=complex)
        self.c3_ = np.zeros((self.nsbl_,self.nflavor_sbl_,self.nflavor_sbl_),dtype=complex)

        #Compute coefficients for fourie transforming G (sign comes from i^n is not included)
        self.c1_G_ = np.identity(self.nflavor_,dtype=complex)
        self.c2_G_ = np.zeros((self.nflavor_,self.nflavor_),dtype=complex)
        self.c3_G_ = np.zeros((self.nflavor_,self.nflavor_),dtype=complex)

    def hyb_freq_to_tau(self, hyb, ntau, beta, ncut):
        nsbl = self.nsbl_
        nflavor_sbl = self.nflavor_sbl_

        hyb_tau=np.zeros((ntau+1,self.nflavor_,self.nflavor_),dtype=complex)
        high_freq_tail=np.zeros((ntau,self.nflavor_,self.nflavor_),dtype=complex)
        hyb_rest=np.zeros((ntau,self.nflavor_,self.nflavor_),dtype=complex)
        matsubara_freq=np.zeros((ntau,),dtype=float)
        tau=np.zeros((ntau+1,),dtype=float)
        for im in range(ntau):
            matsubara_freq[im]=((2*im+1)*np.pi)/beta
        for it in range(ntau+1):
            tau[it]=(beta/ntau)*it
        for isbl in xrange(nsbl):
            start = nflavor_sbl*isbl
            end = nflavor_sbl*(isbl+1)
            tail,rest = ft_to_tau_hyb_with_fitted_tail(ntau, beta, matsubara_freq, tau, self.c1_[isbl,:], hyb[:,start:end,start:end], hyb_tau[:,start:end,start:end])
            high_freq_tail[:,start:end,start:end] = 1.*tail
            hyb_rest[:,start:end,start:end] = 1.*rest
        return hyb_tau, high_freq_tail, hyb_rest

    def G_freq_to_tau(self, G, ntau, beta, ncut):
        G_tau=np.zeros((ntau+1,self.nflavor_,self.nflavor_),dtype=complex)
        matsubara_freq=np.zeros((ntau,),dtype=float)
        tau=np.zeros((ntau+1,),dtype=float)
        for im in range(ntau):
            matsubara_freq[im]=((2*im+1)*np.pi)/beta
        for it in range(ntau+1):
            tau[it]=(beta/ntau)*it
        high_freq_tail,G_rest = ft_to_tau_hyb_with_fitted_tail(ntau, beta, matsubara_freq, tau, self.c1_G_, G, G_tau)
        return G_tau, high_freq_tail, G_rest

def to_freq_bosonic_real_field(ndiv_tau_smpl,beta,n_freq,f_tau,cutoff_rest):
    n_tau_dense = 2*n_freq
    tau_mesh=np.linspace(0,beta,ndiv_tau_smpl+1)
    tau_mesh_dense=np.linspace(0,beta,n_tau_dense+1)
    freq_mesh=np.linspace(0,2*(n_freq-1)*np.pi/beta,n_freq)
    f_freq = np.zeros(n_freq,dtype=complex)

    #Spline interpolation to evaluate the high-frequency tail
    fit = interpolate.InterpolatedUnivariateSpline(tau_mesh,f_tau)
    deriv_0 = fit.derivatives(0.0)
    deriv_beta = fit.derivatives(beta)
    c1 = deriv_beta[0]-deriv_0[0]
    c2 = -(deriv_beta[1]-deriv_0[1])
    c3 = deriv_beta[2]-deriv_0[2]
    for im in range(1,n_freq):
        f_freq[im]=c1/(1J*freq_mesh[im])+c2/(1J*freq_mesh[im])**2+c3/(1J*freq_mesh[im])**3

    #Contribution from the rest part
    f_tau_rest_dense = fit(tau_mesh_dense)-c1*f_tau_tail_bosonic(beta,n_tau_dense,1)-c2*f_tau_tail_bosonic(beta,n_tau_dense,2)-c3*f_tau_tail_bosonic(beta,n_tau_dense,3)
    for im in range(cutoff_rest):
        ftmp=f_tau_rest_dense*np.exp(1J*freq_mesh[im]*tau_mesh_dense[:])
        f_freq[im]+=np.trapz(ftmp,tau_mesh_dense)

    return f_freq

# Utility functions for FT_to_n_bosonic_real_field
#  This returns the Fourier transform of 1/(i omega_m).
def f_tau_tail_bosonic(beta,ndiv_tau,m):
    tau_mesh=np.linspace(0,beta,ndiv_tau+1)
    if m==1:
        return (tau_mesh-beta/2)/beta
    elif m==2:
        return -(0.5/beta)*(tau_mesh-beta/2)**m+beta/24
    elif m==3:
        return (1/(6*beta))*(tau_mesh-beta/2)**m
    else:
        print "Error: m=",m

def to_freq_fermionic_real_field(ndiv_tau_smpl,beta,n_freq,f_tau,cutoff_rest):
    n_tau_dense = 2*n_freq
    tau_mesh=np.linspace(0,beta,ndiv_tau_smpl+1)
    tau_mesh_dense=np.linspace(0,beta,n_tau_dense+1)
    freq_mesh=np.linspace(np.pi/beta, (2*n_freq-1)*np.pi/beta, n_freq)
    f_freq = np.zeros(n_freq,dtype=complex)

    #Spline interpolation to evaluate the high-frequency tail
    fit = interpolate.InterpolatedUnivariateSpline(tau_mesh,f_tau)
    deriv_0 = fit.derivatives(0.0)
    deriv_beta = fit.derivatives(beta)
    c1 = -deriv_beta[0]-deriv_0[0]
    c2 = deriv_beta[1]+deriv_0[1]
    c3 = -deriv_beta[2]-deriv_0[2]
    for im in range(0,n_freq):
        f_freq[im]=c1/(1J*freq_mesh[im])+c2/(1J*freq_mesh[im])**2+c3/(1J*freq_mesh[im])**3

    #Contribution from the rest part
    f_tau_rest_dense = fit(tau_mesh_dense)-c1*f_tau_tail_fermionic(beta,n_tau_dense,1)-c2*f_tau_tail_fermionic(beta,n_tau_dense,2)-c3*f_tau_tail_fermionic(beta,n_tau_dense,3)
    for im in range(cutoff_rest):
        ftmp=f_tau_rest_dense*np.exp(1J*freq_mesh[im]*tau_mesh_dense[:])
        f_freq[im]+=np.trapz(ftmp,tau_mesh_dense)

    return f_freq

# Utility functions for FT_to_n_fermionic_real_field
#  This returns the Fourier transform of 1/(i omega_m).
def f_tau_tail_fermionic(beta,ndiv_tau,m):
    tau_mesh=np.linspace(0,beta,ndiv_tau+1)
    if m==1:
        tau_mesh[:] = -0.5
        return tau_mesh
    elif m==2:
        return 0.25*(-beta+2*tau_mesh)
    elif m==3:
        return 0.25*(beta*tau_mesh-tau_mesh**2)
    else:
        print "Error: m=",m
