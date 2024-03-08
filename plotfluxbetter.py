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

#Condition inputs
name = input('Type the rayleigh number prefix substring')
user_input = input('Type /Full/ for all profile plotting | Type /Flux/ for conv. diff. fluxes:')
# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x, z = dist.local_grids(xbasis, zbasis)
import h5py
filename = "file.hdf5"
filenames = ["profiles_s3.h5",] #delete last profile
filenames = [name + '_' + filename for filename in filenames]
data_dict = dict()
prof = [r"diffusive flux",
r"kinetic energy in x",
r"kinetic energy in z"
,r"mean x velocity"
,r"mean z velocity",
r"mean temperature profile",
r"entrsophy",
r"X diffusive flux",
r"convective flux"]

flux_prof = [r"diffusive flux",
             r"convective flux"]
if user_input == 'Full':
    for task in prof: 
        data_dict[task] = None
    index = 0
    for file in filenames: 
        with h5py.File(path + "/"+name+"_profiles/"+file, "r") as f:
            for task in data_dict.keys():
                data = f['tasks'][task][()].squeeze()
                times = f['scales']['sim_time'][()].squeeze()
                print(times)
                size = np.shape(data)
                index += size[0] 
                task_mean = np.sum(data, axis = 0)
                # print(data.shape())
                if isinstance(data_dict[task], np.ndarray):
                    data_dict[task] += task_mean 
                else:
                    data_dict[task] = task_mean 
    for task in data_dict.keys():
        plt.plot(z.squeeze(), data_dict[task]/(index))
        plt.xlabel('z')
        plt.ylabel(task)
        plt.title(task)
        plt.savefig(path+"/"+name+"_profiles/"+task+'_fig.png')
        plt.close()            


if user_input == "Flux":
    for task in flux_prof: 
        data_dict[task] = None
    index = 0
    for file in filenames:
        with h5py.File(path + "/"+name+"_profiles/"+file, "r") as f:
            for task in data_dict.keys():
                data = f['tasks'][task][()].squeeze()
                print(np.shape(data))
                times = f['scales']['sim_time'][()].squeeze()
                print(times)
                size = np.shape(data)
                print("size = {}".format(size))
                index += size[0] 
                task_mean= np.sum(data, axis = 0)
                # print(data.shape())
                if isinstance(data_dict[task], np.ndarray):
                    data_dict[task] += task_mean 
                else:
                    data_dict[task] = task_mean

    plt.plot(z.squeeze(),data_dict['convective flux']/(index), label = 'Convective flux')
    plt.plot(z.squeeze(),data_dict['diffusive flux']/(index), label = 'Diffusive flux')
    tot_fluxmean =data_dict['convective flux']/(index)+data_dict['diffusive flux']/(index)
    plt.plot(z.squeeze(),tot_fluxmean, label = 'Total flux')
    plt.legend(loc = 'upper right')
    plt.xlabel('z')
    plt.ylabel("Flux")
    plt.title("Heat Fluxes Plot")
    file_name = path+"/"+name+"_profiles/"+"Flux_fig.png"
    plt.savefig(file_name)
    print(file_name)
    plt.close()
                

