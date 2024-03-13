import os 
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import sys
path = os.path.dirname(os.path.abspath(__file__))
print(path) #independent
Lx, Lz = float(1), 1
n = 6#power of two 
Nz_prime = 2**n
Nx_prime = float(1) * Nz_prime
Nx, Nz = Nx_prime, Nz_prime #Nx, Nz = 1024, 256 #4Nx:Nz locked ratio~all powers of 2 Nx, Nz = Nx_prime, Nz_prime
Rayleigh = float(2e5) #CHANGEABLE/Take Notes Lower Number~More turbulence resistant Higher Number~Less turbulence resistant
Prandtl = float(1)
dealias = 3/2
stop_sim_time = float(80)
timestepper = d3.RK222
max_timestep = 0.1
dtype = np.float64
#Condition inputs
name = input('Type the rayleigh number prefix substring')
user_input = input('Type /Full/ for all profile plotting | Type /Flux/ for conv. diff. fluxes:')
startinput = input("Please give start profile number:")
stopinput = input("Please give stop profile number:")
# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x, z = dist.local_grids(xbasis, zbasis)
import h5py
filename = "file.hdf5"
# filenames = ["profiles_s11.h5","profiles_s10.h5","profiles_s9.h5","profiles_s8.h5","profiles_s7.h5","profiles_s6.h5","profiles_s5.h5","profiles_s4.h5","profiles_s3.h5","profiles_s2.h5","profiles_s1.h5"] #delete last profile
filenames = ["profiles_s{}.h5".format(i) for i in range(startinput, stopinput+1)]
filenames = [name + '_' + filename for filename in filenames]
print("CHECK FILENAMES")
data_dict = dict()
prof = [r"diffusive flux",
r"kinetic energy in x",
r"kinetic energy in z"
,r"mean x velocity"
,r"mean z velocity",
r"mean temperature profile",
r"entrsophy",
r"X diffusive flux",
r"convective flux",r"reynolds"]
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
    plotornot = input("Please type what function you would like to perform. Type /Plot/ to plot | Type /Archive/ to save data to external file:")
    if plotornot == "Plot":
        for task in data_dict.keys():
            plt.plot(z.squeeze(), data_dict[task]/(index))
            plt.xlabel('z')
            plt.ylabel(task)
            plt.title(task+ " at Ra= "+name)
            plt.savefig(path+"/"+name+"_profiles/"+task+'_fig.png')
            plt.close()
    elif plotornot == "Archive":
        archname = input("Please provide conditions of simulation. Type /Bump/ if simulation was run with a present conductivity bump | Leave blank if no bump was present:")
        if archname == "Bump":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"_profiles/"+task+'_dataBUMP.csv'), "a")
                        savef.write(str(val))
                        savef.write("\n")         
        if archname == "":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"_profiles/"+task+'_dataBUMP.csv'), "a")
                        savef.write(str(val))
                        savef.write("\n")  
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
    plotornot = input("Please type what function you would like to perform. Type /Plot/ to plot | Type /Archive/ to save data to external file:")
    if plotornot == "Plot":
        for task in data_dict.keys():
            plt.plot(z.squeeze(),data_dict['convective flux']/(index), label = 'Convective flux')
            plt.plot(z.squeeze(),data_dict['diffusive flux']/(index), label = 'Diffusive flux')
            tot_fluxmean =data_dict['convective flux']/(index)+data_dict['diffusive flux']/(index)
            plt.plot(z.squeeze(),tot_fluxmean, label = 'Total flux')
            plt.legend(loc = 'upper right')
            plt.xlabel('z')
            plt.ylabel("Flux")
            plt.title("Heat Fluxes Plot Ra= "+name)
            file_name = path+"/"+name+"_profiles/"+name+"_Flux_fig.png"
            plt.savefig(file_name)
            print(file_name)
            plt.close()
    elif plotornot == "Archive":
        archname = input("Please provide conditions of simulation. Type /Bump/ if simulation was run with a present conductivity bump | Leave blank if no bump was present:")
        if archname == "Bump":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"_profiles/"+task+'_dataBUMP.csv'), "a")
                        savef.write(str(val))
                        savef.write("\n")         
        if archname == "":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"_profiles/"+task+'_dataBUMP.csv'), "a")
                        savef.write(str(val))
                        savef.write("\n")


