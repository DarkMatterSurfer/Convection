import numpy as np
import matplotlib.pyplot as plt 
import scipy.integrate as integrate
import scipy.special as special
import os
import sys
sig = 0.001
pi = np.pi
N = 1000
coeff_indexes = np.linspace(1,N,(N-1)+1)
cos_list = []
def coeff_integralcos(index):
    coeff_cos = integrate.quad(lambda x: (2*np.exp((-(x-1/2)**2)/(2*(sig**2)))*np.cos(2*pi*index*x)), 0, 1)
    return coeff_cos
for i in range(len(coeff_indexes)):
   cos_coeff = coeff_integralcos(coeff_indexes[i])
   cos_list.append(cos_coeff)
# print(cos_list)

for i in range(len(cos_list)):
    cos_list[i]=cos_list[i][0]
# print(cos_list)

sin_list = []
def coeff_integralsin(index):
    coeff_sin = integrate.quad(lambda x: (2*np.exp((-(x-1/2)**2)/(2*(sig**2)))*np.sin(2*pi*index*x)), 0, 1)
    return coeff_sin
for i in range(len(coeff_indexes)):
   sin_coeff = coeff_integralsin(coeff_indexes[i])
   sin_list.append(sin_coeff)
# print(sin_list)

for i in range(len(sin_list)):
    sin_list[i]=sin_list[i][0]
print(sin_list)

domain_X = np.linspace(0,1,200)
f = np.zeros_like(domain_X)
for index,coeff in enumerate(cos_list):
    f+=coeff*np.cos(2*pi*(index+1)*domain_X)
plt.plot(domain_X,f)
plt.show()
plt.close