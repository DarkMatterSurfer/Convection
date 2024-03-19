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
user_input = "Combined"
#input('Type /Full/ for all profile plotting | Type /Flux/ for conv. diff. fluxes:')
if user_input == "Full" or user_input == "Flux":
    startinput = input("Please give start profile number:")
    stopinput = input("Please give stop profile number:")
    filename = "file.hdf5"
    # filenames = ["profiles_s11.h5","profiles_s10.h5","profiles_s9.h5","profiles_s8.h5","profiles_s7.h5","profiles_s6.h5","profiles_s5.h5","profiles_s4.h5","profiles_s3.h5","profiles_s2.h5","profiles_s1.h5"] #delete last profile
    filenames = ["profiles_s{}.h5".format(i) for i in range(int(startinput), int(stopinput)+1)]
    print(filenames)

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x, z = dist.local_grids(xbasis, zbasis)


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
        with h5py.File(path + "/"+name+"/profiles/"+file, "r") as f:
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
            Rayleigh = config.getfloat('param', 'Ra')
            plt.title("Heat Fluxes Plot Ra= "+ str(Rayleigh))
            plt.savefig(path+"/"+name+"/"+task+'_fig.png')
            plt.close()
    if plotornot == "Archive":
        archname = "testofscript"
        #input("Please provide conditions of simulation. Type /Bump/ if simulation was run with a present conductivity bump | Leave blank if no bump was present:")
        if archname == "Bump":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"/"+task+'_dataBUMP.csv'), "a")
                        savef.write(str(val))
                        savef.write("\n")         
        if archname == "":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"/"+task+'_data.csv'), "a")
                        savef.write(str(val))
                        savef.write("\n")  
if user_input == "Flux":
    for task in flux_prof: 
        data_dict[task] = None
    index = 0
    for file in filenames:
        with h5py.File(path + "/"+name+"/profiles/"+file, "r") as f:
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
            Rayleigh = config.getfloat('param', 'Ra')
            plt.title("Heat Fluxes Plot Ra= "+ str(Rayleigh))
            file_name = path+"/"+name+"/"+"Flux_fig.png"
            plt.savefig(file_name)
            print(file_name)
            plt.close()
    #Making csv files for profiles
    if plotornot == "Archive":
        archname = "testofscript"
        #input("Please provide conditions of simulation. Type /Bump/ if simulation was run with a present conductivity bump | Leave blank if no bump was present:")
        if archname == "Bump":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"/"+task+'_dataBUMP.csv'), "a")
                        savef.write(str(val))
                        savef.write("\n")         
        if archname == "":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"/"+task+'_data.csv'), "a")
                        savef.write(str(val))
                        savef.write("\n")
if user_input == "Combined":
    nobump = []
    bump = []
    nobump_archive = path+"/"+name+"/convective flux_data.csv"
    bump_archive = path+"/"+name+"/convective flux_dataBUMP.csv"
    archive_data = [nobump_archive,bump_archive]
    print(range(0,len(archive_data)-1))
    for i in range(0,len(archive_data)):
        if i == 0:
            #Read no bump csv convection file
            with open(archive_data[i]) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    nobump.append(row[0])
            print("This is no bump data for Ra="+str(Rayleigh)+":\n\n\n", nobump)
            print("\n\n")
        if i == 1:
            #Read bump csv convection file
            with open(archive_data[i]) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    bump.append(row[0])
            print("This is bump data for Ra="+str(Rayleigh)+" :\n\n\n", bump)
            print("\n\n")
    print(type(nobump[0]))
    print(type(bump[0]))
    #Plotting
    plt.plot(z.squeeze(),nobump, label = 'Norm.')
    plt.plot(z.squeeze(),bump, label = 'Bump')
    plt.title("Combined Convective Flux Plot Ra= "+str(Rayleigh))
    plt.legend(loc = 'upper right')
    plt.xlabel('z')
    plt.ylabel("Flux")
    file_name = path+"/"+name+"/"+"CombinedConvective_fig.png"
    plt.savefig(file_name)
    plt.show()