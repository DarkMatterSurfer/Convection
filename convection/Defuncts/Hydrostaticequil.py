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
tau_p = dist.Field(name = 'tau_p')
tau_g = dist.Field(name = 'tau_g')
tau_T1 = dist.Field(name = 'tau_T1')
tau_T2 = dist.Field(name = 'tau_T2')

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
dz = lambda A: d3.Differentiate(A, coords['z'])
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
#Problemas
problem = d3.NLBVP([rho, p, s, T, g, tau_p, tau_g, tau_T1, tau_T2], namespace=locals())
problem.add_equation("dz(p)+ lift(tau_p) = -rho*g")
problem.add_equation("dz(-g)-4*pi*G*rho + lift(tau_g) = 0")
problem.add_equation("dz(k*dz(T)+ lift(tau_T1)) + lift(tau_T2)  = -Q ")
problem.add_equation("p = K*rho**(gamma)")
problem.add_equation("s = c_v * np.log((p)/(rho**gamma))")
#Boundary Conditon
problem.add_equation("p(z=0) = Lz")
problem.add_equation("g(z=0) = 0")
problem.add_equation("rho(z=0) = 0")
problem.add_equation("T(z=0) = 0")
#problem.add_equation("integ(p) = 0")
#Solver 
solver = problem.build_solver()
solver.solve()