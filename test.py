import numpy as np
import fourier_transform as ft

nf = 2

beta = 10.0
ntau = 10
E = np.array([0.2, 0.5])
c1 = np.array([[1.0, 0.0],[0.0,2.0]],dtype=complex)
c2 = np.array([[0.2, 0.1+0.1J],[0.1-0.1J,0.5]],dtype=complex)
c3 = 0.1*c2

theta = 0.2*np.pi
c = np.cos(theta)
s = np.sin(theta)
Umat = np.array([[c, -s],[s,c]],dtype=complex)
Umat[:,0] *= np.exp(0.4J*np.pi)

omega_n = np.array([(2*n+1)*np.pi/beta for n in xrange(ntau)])
tau = np.linspace(0,beta,ntau+1)

Gomega = np.zeros((ntau,nf,nf),dtype=complex)
Gtau = np.zeros((ntau+1,nf,nf),dtype=complex)

for im in xrange(ntau):
    Gomega[im,:,:] = c1/(1J*omega_n[im])+c2/((1J*omega_n[im])**2)+c3/((1J*omega_n[im])**3)

#for i in xrange(ntau+1):
    #Gtau[i,:,:] = np.dot(np.dot(Umat.conjugate().transpose(),Gtau[i,:,:]),Umat)

#for i in xrange(ntau):
    #Gomega[i,:,:] = np.dot(np.dot(Umat.conjugate().transpose(),Gomega[i,:,:]),Umat)

#c1 = np.diag([1+0.0J]*nf)
#c2 = np.zeros((nf,nf),dtype=complex)
#c3 = np.zeros((nf,nf),dtype=complex)
#
#Gtau_fft = np.zeros_like(Gtau)
#ft.ft_to_tau_hyb(ntau, beta, omega_n, tau, c1, c2, c3, Gomega, Gtau_fft, ntau/2)
#
#Gomega_fft = ft.G_tau_to_freq2(ntau,beta,Gtau,ntau/2)
#
#f = open('G.txt', 'w')
#for flavor in xrange(nf):
   #for flavor2 in xrange(nf):
      #for i in xrange(ntau+1):
         #print>>f, i, Gtau[i,flavor,flavor2].real, Gtau[i,flavor,flavor2].imag
      #print>>f, ""
      #print>>f, ""
#f.close()
#
#f = open('G-fft.txt', 'w')
#for flavor in xrange(nf):
   #for flavor2 in xrange(nf):
      #for i in xrange(ntau+1):
         #print>>f, i, Gtau_fft[i,flavor,flavor2].real, Gtau_fft[i,flavor,flavor2].imag
      #print>>f, ""
      #print>>f, ""
#f.close()
#
#f = open('Gomega.txt', 'w')
#for flavor in xrange(nf):
   #for flavor2 in xrange(nf):
      #for i in xrange(ntau):
         #print>>f, i, Gomega[i,flavor,flavor2].real, Gomega[i,flavor,flavor2].imag
      #print>>f, ""
      #print>>f, ""
#f.close()
#

#f = open('Gomega-fft.txt', 'w')
#for flavor in xrange(nf):
   #for flavor2 in xrange(nf):
      #for i in xrange(ntau):
         #print>>f, i, Gomega_fft[i,flavor,flavor2].real, Gomega_fft[i,flavor,flavor2].imag
      #print>>f, ""
      ##print>>f, ""
#f.close()

##### TEST ESTIMATE TAIL ######
c1_fit, c2_fit, c3_fit = ft.estimate_tail(ntau, beta, omega_n, tau, c1, Gomega)
#print c3
#print c3_fit
print np.abs(c1-c1_fit)
print np.abs(c2-c2_fit)
print np.abs(c3-c3_fit)
ft.ft_to_tau_hyb_with_fitted_tail(ntau, beta, omega_n, tau, c1, Gomega, Gtau)
f = open('G.txt', 'w')
for flavor in xrange(nf):
   for flavor2 in xrange(nf):
      for i in xrange(ntau+1):
         print>>f, i, Gtau[i,flavor,flavor2].real, Gtau[i,flavor,flavor2].imag
      print>>f, ""
      print>>f, ""
f.close()
