from docopt import docopt
from configparser import ConfigParser
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import mpi4py
import matplotlib.pyplot as plt
from docopt import docopt
import sys
import os 
path = os.path.dirname(os.path.abspath(__file__))
# configfile = path +"/options.cfg"
# args = docopt(__doc__)
if len(sys.argv) < 2:
    print('please provide config file')
    raise
else:
    configfile = sys.argv[1]

config = ConfigParser()
config.read(str(configfile))
CW = mpi4py.MPI.COMM_WORLD
ncores = CW.size
# Parameters
Lx, Lz = config.getfloat('param', 'Lx'), config.getfloat('param', 'Lz')
n = config.getint('param', 'n') #power of two 
Nz_prime = 2048
Nx_prime = config.getint('param','aspect') * Nz_prime #float(args['--lx']) 
Nx, Nz = Nx_prime, Nz_prime
print((Nx,Nz))#Nx, Nz = 1024, 256 #4Nx:Nz locked ratio~all powers of 2 Nx, Nz = Nx_prime, Nz_prime
Rayleigh = config.getfloat('param', 'Ra') #CHANGEABLE/Take Notes Lower Number~More turbulence resistant Higher Number~Less turbulence resistant
Prandtl = config.getfloat('param', 'Pr')
dealias = 3/2
stop_sim_time = config.getfloat('param', 'st')
timestepper = d3.RK222
max_timestep = config.getfloat('param', 'maxtimestep')
dtype = np.float64
name = config.get('param', 'name')
koopa1D= config.getboolean('param','koopa1D')
# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x, z = dist.local_grids(xbasis, zbasis)
# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
integx = lambda arg: d3.Integrate(arg, 'x')
integ = lambda arg: d3.Integrate(integx(arg), 'z')
temperature = 2*(z-0.5)
b = temperature
# Hollow Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
def hollowsub(sig,e,Tbump,A):
    Tplus = b -Tbump + e
    Tminus = b -Tbump - e
    pi = np.pi
    koopa = kappa*A*(((-pi/2)+np.arctan(sig*Tplus*Tminus))/((pi/2)+np.arctan(sig*e*e)))
    return koopa
plt.figure(figsize=(8,5))
plt.plot((b).ravel(),(kappa+hollowsub(30,0.1,0,0.5)).ravel(), color='k', linestyle='solid', linewidth=2, label = "A = 0.5")
plt.plot((b).ravel(),(kappa+hollowsub(30,0.1,0,0.7)).ravel(), color='g', linestyle='dotted', linewidth=2, label = "A = 0.7")
plt.plot((b).ravel(),(kappa+hollowsub(30,0.1,0,0.3)).ravel(), color='b', linestyle='dashdot', linewidth=2, label = "A = 0.3")
plt.axhline(y = 0.0010, color = 'r', linestyle = '--',linewidth=1, label = "A = 0.0") 
plt.legend(loc = "lower right", prop = {'size':10})
plt.ylabel(r"$\kappa(b)$")
plt.xlabel("b")
plt.title("Thermal Diffusivity for Decrease Amplitudes")
plt.xlim(-1.0,1.0)
plt.ylim(0.0,0.0012)
print(path+"/bumpfunction.png")
plt.savefig(path+"/bumpfunction.png")
