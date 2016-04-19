import sys
#import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros
from math import sin, cos, pi, sqrt

def get_bounds(GRID, BINS, U, Vnn):
  cos_ = []
  for i in range(-GRID,GRID):
    k = pi*i/GRID
    cos_.append(cos(k))

  vals = []
  for x in range(0,GRID+1):
    C = U+2*Vnn*cos_[x]
    for y in range(x,GRID+1):
      D = C + 2*Vnn*cos_[y]
      for z in range(y,GRID+1):
        rtmp = D + 2*Vnn*cos_[z]
        if rtmp != 0.0:
          vals.append(1/rtmp)
  #print vals
  vals = np.array(vals)

  if len(vals)>0:
      return np.amin(vals), np.amax(vals)
  else:
      return 0.0, 0.01


def gen_ivk_dos(GRID, BINS, U, Vnn, lower, upper):
  bin_width = (upper-lower) / BINS
  DOS = zeros(BINS+2)

  def increment(e,m):
    itmp = int(round((e - lower)/bin_width))
    if itmp>=0 and itmp<len(DOS):
      DOS[int(round((e - lower)/bin_width))] += m

  cos_ = []
  for i in range(-GRID,GRID):
    k = pi*i/GRID
    cos_.append(cos(k))

  def Multiplicity(x,y,z):
    # for sorted 0<=x<=y<=z<=GRID gives the "mirror"-multiplicity
    if x>0 and z<GRID:
      return 8
    q=0
    if x==0 or x==GRID:
      q+=1
    if y==0 or y==GRID:
      q+=1
    if z==0 or z==GRID:
      q+=1
    if q==0:
      return 8
    elif q==1:
      return 4
    elif q==2:
      return 2
    else:
      return 1

  def CompleteMultiplicity(x,y,z):
    # for ordinarily sorted 0<=x<=y<=z<=GRID gives the multiplicity
    if x<y and y<z:
      return 6*Multiplicity(x,y,z)
    else:
      q=0
      if x==y:
        q+=1
      if y==z:
        q+=1
      if q==1:
        return 3*Multiplicity(x,y,z)
      else:
        return Multiplicity(x,y,z)

  for x in range(0,GRID+1):
    C = U+2*Vnn*cos_[x]
    for y in range(x,GRID+1):
      D = C + 2*Vnn*cos_[y]
      for z in range(y,GRID+1):
        rtmp = D + 2*Vnn*cos_[z]
        if rtmp != 0.0:
          increment( 1/rtmp, CompleteMultiplicity(x,y,z))

  counter=0
  inc = 1./(8.*GRID*GRID*GRID*bin_width)  # normalized to 1
  for x in range(0,len(DOS)):
    counter+=DOS[x]
    DOS[x]*=inc
  #print "Number of processed k-points: ",counter, "  (should be ", 8*GRID*GRID*GRID,')'
  
  # correct normalization for the 1st and last bin
  #DOS[0] *= 2.      # it is a half-bin (has only half of the usual width)
  #DOS[BINS] *= 2.   # it is a half-bin (has only half of the usual width)

  energies = zeros(BINS+2)
  energies[0] = lower
  energies[BINS] = upper
  for j in range(0,BINS):
    energies[j] = lower + j*bin_width

  def func(x,n):
    if n==0:
      return DOS[x]
    elif n==1:
      return DOS[x]*energies[x]
    else:
      return DOS[x]*energies[x]*energies[x]

  def Integrate(n):
    if ( BINS % 2 == 1 ):
      return " error: for Simpson integration use even number of bins"
    sum1 = 0
    sum2 = 0
    halfstep = bin_width
  
    for i in range(1, BINS-1, 2):
      sum2 += func(i,n);
      sum1 += func(i+1,n);
  
    sum1 = 2. * sum1 + 4. * (sum2+func(BINS-1,n)) + func(0,n) + func(BINS,n);
    return sum1 * halfstep / 3.

  #print "Checks:"
  norm = Integrate(0)
  #print "  normalization = ", norm,"  (close to 1)"
  #print "  first moment of the normalized DOS = ", Integrate(1)/norm,"  (exact: 0.0)"
  #print "  second moment of the normalized DOS = ", Integrate(2)/norm,"  (exact: 6.0)"
  #
  #print "Histogram created."

  #plt.plot(energies[0:BINS+1],DOS[0:BINS+1],'r-')      
  #plt.xlabel('energy / t --->')
  #plt.ylabel('DOS  --->')
  #plt.title('DOS of the cubic lattice')
  #plt.show()

  r = np.zeros((2,BINS+1),dtype=float)
  r[0,:] = 1.0*energies[0:BINS+1]
  r[1,:] = 1.0*DOS[0:BINS+1]
  return r

  #print "Do you wish to save the histogram [y/n] ?"
  #answer = 'y'

  #if answer[0]=='y':
    ## write into file
    #print "Set the name for the histogram output file:"
    #file_name = raw_input('--> ')
    #file_out = open(file_name,'w')
    #for j in range(0, BINS+1):
      #file_out.write(str(energies[j]))
      #file_out.write('  ')
      #file_out.write(str(DOS[j]))
      #file_out.write('\n')
    #file_out.close()
