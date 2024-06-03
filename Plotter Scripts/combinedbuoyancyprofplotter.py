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
import h5py
import csv
path = os.path.dirname(os.path.abspath(__file__))
print(path) #independent
Lx, Lz = float(1), 1
if len(sys.argv) < 2:
    print('please provide config file')
    raise
else:
    configfile = sys.argv[1]

config = ConfigParser()
config.read(str(configfile))
# Parameters
Lx, Lz = config.getfloat('param', 'Lx'), config.getfloat('param', 'Lz')
n = config.getint('param', 'n') #power of two 
Nz_prime = 2**n
Nx_prime = 4 * Nz_prime #float(args['--lx']) 
Nx, Nz = Nx_prime, Nz_prime #Nx, Nz = 1024, 256 #4Nx:Nz locked ratio~all powers of 2 Nx, Nz = Nx_prime, Nz_prime
Rayleigh = config.getfloat('param', 'Ra') #CHANGEABLE/Take Notes Lower Number~More turbulence resistant Higher Number~Less turbulence resistant
Prandtl = config.getfloat('param', 'Pr')
dealias = 3/2
stop_sim_time = config.getfloat('param', 'st')
timestepper = d3.RK222
max_timestep = config.getfloat('param', 'maxtimestep')
dtype = np.float64
name = config.get('param', 'name')
#Condition inputs


# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x, z = dist.local_grids(xbasis, zbasis)


print("CHECK FILENAMES")
data_dict = dict()
prof = [r"diffusiveflx",
r"kex",
r"kez"
,r"mean_u@x"
,r"mean_u@z"]
flux_prof = [r"diffusiveflx",
             r"convectiveflx"]


tempA0 = []
tempA0p3 = []
tempA0p5 = []
tempA0p7 = []
tempA0archive = path+"/"+"1e6DATAnobump"+r"/mean_temp_data.csv"
A0p5archive = path+"/"+"1e6bump_try2"+r"/mean_temp_dataBUMP.csv"
A0p3archive = path+"/"+"A0p3Ra1e6"+r"/mean_temp_dataBUMP.csv"
A0p7jarchive = path+"/"+"A0p7Ra1e6"+r"/mean_temp_dataBUMP.csv"
#path+"/"+"le6bump_try2"+r"/convectiveflx_dataBUMP.csv"
archive_data = [tempA0archive,A0p5archive,A0p3archive,A0p7jarchive]
print(range(0,len(archive_data)-1))
for i in range(0,len(archive_data)):
    if i == 0:
        #Read no bump csv convection file
        # with archive_data[i] as csvfile:
        tempA0 = np.loadtxt(archive_data[i], delimiter=None, dtype=float)
        print("This is no bump data for Ra="+str(Rayleigh)+":\n\n\n", tempA0)
        print("\n\n")
    if i == 1:
        #Read bump csv convection file
        tempA0p5 = np.loadtxt(archive_data[i], delimiter=None, dtype=float)
        print("This is bump data for Ra="+str(Rayleigh)+" :\n\n\n",tempA0p5 )
        print("\n\n")
    if i == 2:
        #Read effective csv convection file
        tempA0p3 = np.loadtxt(archive_data[i], delimiter=None, dtype=float)
        print("This is effective rayleigh data for Ra="+str(Rayleigh/0.5)+" :\n\n\n", tempA0p3)
        print("\n\n")
    if i == 3:
        tempA0p7 = np.loadtxt(archive_data[i], delimiter=None, dtype=float)
        print("This is horizontally averaged data for Ra="+str(Rayleigh/0.5)+" :\n\n\n", tempA0p7)
        print("\n\n")
    # print(type(nobump[0]))
    # print(type(bump[0]))
    #Plotting


plt.plot(z.squeeze(),tempA0/Lx, label = 'A = 0')
plt.plot(z.squeeze(),tempA0p5/Lx,"r:", linewidth = 3, markersize= 3,label = 'A = 0.5')
x, z = dist.local_grids(xbasis, zbasis,scales = 2)
plt.plot(z.squeeze(),tempA0p7/Lx,"ko", markersize = 3, alpha = 0.25, label = 'A = 0.7')
plt.plot(z.squeeze(),tempA0p3/Lx,"g--",label="A = 0.3") 
plt.title("Combined Mean Temperature Profile Nominal " + "Ra = 1e6" )
plt.legend(loc = 'upper right')
plt.xlabel('z')
plt.ylabel(r"$b_{mean}$")
plt.savefig(path+"/tempprofileRA1e6.png")