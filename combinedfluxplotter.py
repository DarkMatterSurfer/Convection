import os 
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import sys
path = os.path.dirname(os.path.abspath(__file__))
print(path) #independent
Lx, Lz = float(1), 1
n = 6 #power of two 
Nz_prime = 2**n
Nx_prime = float(1) * Nz_prime
Nx, Nz = Nx_prime, Nz_prime #Nx, Nz = 1024, 256 #4Nx:Nz locked ratio~all powers of 2 Nx, Nz = Nx_prime, Nz_prime
Rayleigh = float(2e6) #CHANGEABLE/Take Notes Lower Number~More turbulence resistant Higher Number~Less turbulence resistant
Prandtl = float(1)
dealias = 3/2
stop_sim_time = float(150)
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x, z = dist.local_grids(xbasis, zbasis)
import h5py
filename = "file.hdf5"
prof= [r"diffusive flux",
        r"kinetic energy in x",
        r"kinetic energy in z"
        ,r"mean x velocity"
        ,r"mean z velocity",
        r"entrsophy",
        r"X diffusive flux",
        r"convective flux"]
with h5py.File(path + '/profiles/profiles_s7.h5', "r") as f:
    def figmaker(task,start,end):
        #data = f['tasks'][task][()].squeeze()
        conv = f['tasks'][r"convective flux"][()].squeeze()
        diff = f['tasks'][r"diffusive flux"][()].squeeze()
        # print(data.shape())
        #task_data = data[start:end,:]
        conv_data = conv[start:end,:]
        diff_data = diff[start:end,:]
        # print(task_data.shape())
        #task_mean = np.mean(task_data, axis = 0)
        conv_mean = np.mean(conv_data, axis = 0)
        diff_mean = np.mean(diff_data, axis = 0)
        # print(task_mean.shape())
        plt.plot(z.squeeze(),conv_mean+diff_mean, label = 'Total Flux')
        plt.plot(z.squeeze(),diff_mean, label = 'Diffusive Flux')
        plt.plot(z.squeeze(),conv_mean, label = 'Convective Flux')
        plt.legend(loc = 'upper right')
        plt.xlabel('z')
        plt.ylabel('Flux Amplitude')
        plt.title('FLux')
        plt.savefig(path+"/profiles/"+'Flux_fig.png')
        plt.close()
    #for i in list:
        #figmaker(i,0,-1)
    for i in prof:
            figmaker(i,0,-1)