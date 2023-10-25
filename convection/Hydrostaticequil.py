import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import mpi4py
import matplotlib.pyplot as plt
from docopt import docopt
import sys

Lz = 1 
Nz = 128
dtype = np.float64 #regular vs complex128 regular gives real numbers
#coords
coords = d3.CartesianCoordinates('z') 
dist = d3.Distributor(coords, dtype=dtype)
zbasis = d3.Chebyshev(coords['z'], size = Nz, bounds=(0, Lz))
# FIelds 
rho = dist.Field(name = 'rho', bases = (zbasis,)) #density
p = dist.Field(name = "p", bases =(zbasis,)) # 
s = dist.Field(name = 's', bases = (zbasis,))
T = dist.Field(name = 'T', bases = (zbasis,))
g = dist.Field(name = 'g', bases = (zbasis,))

k = dist.Field(name = 'k', bases = (zbasis,)) # therm diff
Q = dist.Field(name = 'Q', bases = (zbasis,)) # added heat

k['g'] = 1
Q['g'] = 1
K = 1
gamma = 4/3
c_v = 1
pi = np.pi 
G = 1
z = dist.local_grids(zbasis)

#Problemas
problem = d3.LBVP([rho, p, s, T, g], namespace=locals())
problem.add_equation("dz(p) = -rho*g")
problem.add_equation("dz(-g)-4*pi*G*rho = 0")
problem.add_equation("dz(k*dz(T)) = -Q ")
problem.add_equation("p = K*rho**(gamma)")
problem.add_equation("s = c_v * log((p)/(rho**gamma))")