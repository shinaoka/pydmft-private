import numpy as np
from numpy import cos,pi
from numpy.linalg import inv
import sys
import time
import re 
import os
import subprocess

class DMFTResult:
    def update(self, Hk_mean):
        self.Hk_mean = Hk_mean

dmft_result = DMFTResult()
dmft_result.update(np.zeros((2,2)))

print "debug", dmft_result.Hk_mean
print "debug", vars(dmft_result)

dict = vars(dmft_result)
#print type(dict)
#print dict["Hk_mean"], len(dict)
for k,v in vars(dmft_result).items():
    print k,v
