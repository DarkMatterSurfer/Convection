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
name = "A0p7Ra1e6" #config.get('param', 'name')
#Condition inputs
user_input = input('Type /Full/ for all profile plotting | Type /Flux/ for conv. diff. fluxes:')
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
prof = [
# r"diffusiveflx",
# r"kex",
# r"kez"
# ,r"mean_u@x"
# ,r"mean_u@z",
 r"mean temperature profile",
# r"entrsophy",
# r"Xdiff",
# r"convectiveflx",r"reynolds"
]
flux_prof = [r"diffusiveflx",
             r"convectiveflx"]
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
            title_input = input("Provide string literal for plot for "+task+"plot:")
            if title_input == "":
                plt.title(task+" Plot Ra= "+ str(Rayleigh))
            else:
                plt.title(title_input+" Plot Ra= "+ str(Rayleigh))
            plt.savefig(path+"/"+name+"/"+task+'_fig.png')
            plt.close()
    if plotornot == "Archive":
        archname = input("Please provide conditions of simulation. Type /Bump/ if simulation was run with a present conductivity bump | Leave blank if no bump was present:")
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
            plt.plot(z.squeeze(),data_dict['convectiveflx']/(index), label = 'Convective flux')
            plt.plot(z.squeeze(),data_dict['diffusiveflx']/(index), label = 'Diffusive flux')
            tot_fluxmean =data_dict['convectiveflx']/(index)+data_dict['diffusiveflx']/(index)
            plt.plot(z.squeeze(),tot_fluxmean, label = 'Total flux')
            plt.legend(loc = 'upper right')
            plt.xlabel('z')
            plt.ylabel("Flux")
            Rayleigh = config.getfloat('param', 'Ra')
            title=input("Give title /Title of Heat Fluxes/:")
            plt.title(title+" for Ra= "+ str(Rayleigh))
            file_name = path+"/"+name+"/"+"Flux_fig.png"
            plt.savefig(file_name)
            print(file_name)
            plt.close()
    #Making csv files for profiles
    if plotornot == "Archive":
        archname = input("Please provide conditions of simulation. Type /Bump/ if simulation was run with a present conductivity bump | Leave blank if no bump was present:")
        if archname == "Bump":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"/"+task+'_dataBUMP.csv'), "a")
                        savef.write(str(val/index))
                        savef.write("\n")         
        if archname == "":
            for task in data_dict.keys():
                    for val in np.nditer(data_dict[task].T, order='C'): 
                        savef = open((path+"/"+name+"/"+task+'_data.csv'), "a")
                        savef.write(str(val/index))
                        savef.write("\n")
if user_input == "Combined":
    nobump = []
    bump = []
    eff = []
    horiz_bump = []
    nobump_archive = path+"/"+"1e6nobump"+r"/convective flux_data.csv"
    bump_archive = "/home/brogers/Convection/1e6bump_try2/convectiveflx_dataBUMP.csv" 
    #path+"/"+"le6bump_try2"+r"/convectiveflx_dataBUMP.csv"
    effective_archive = path+"/"+"1e6effective"+r"/convectiveflx_data.csv"
    horiz_archive = path +"/"+name+r"/convectiveflx_dataBUMP.csv"
    archive_data = [nobump_archive,bump_archive,effective_archive,horiz_archive]
    print(range(0,len(archive_data)-1))
    for i in range(0,len(archive_data)):
        if i == 0:
            #Read no bump csv convection file
            # with archive_data[i] as csvfile:
            nobump = np.loadtxt(nobump_archive, delimiter=None, dtype=float)
            print("This is no bump data for Ra="+str(Rayleigh)+":\n\n\n", nobump)
            print("\n\n")
        if i == 1:
            #Read bump csv convection file
            bump = np.loadtxt(bump_archive, delimiter=None, dtype=float)
            print("This is bump data for Ra="+str(Rayleigh)+" :\n\n\n", bump)
            print("\n\n")
        if i == 2:
            #Read effective csv convection file
            eff = np.loadtxt(effective_archive, delimiter=None, dtype=float)
            print("This is effective rayleigh data for Ra="+str(Rayleigh/0.5)+" :\n\n\n", bump)
            print("\n\n")
        if i == 3:
            horiz_bump = np.loadtxt(horiz_archive, delimiter=None, dtype=float)
            print("This is horizontally averaged data for Ra="+str(Rayleigh/0.5)+" :\n\n\n", bump)
            print("\n\n")
    # print(type(nobump[0]))
    # print(type(bump[0]))
    #Plotting
    nomRa = input("Nominal Ra:")
    plt.plot(z.squeeze(),nobump, label = 'Norm.')
    plt.plot(z.squeeze(),eff,"g--",label="Eff. Ra= 2e6")
    # x, z = dist.local_grids(xbasis, zbasis,scales = 2)
    plt.plot(z.squeeze(),bump,"r:", linewidth = 3, markersize= 3,label = 'Bump')
    plt.plot(z.squeeze(),horiz_bump,"yo", markersize = 2, alpha = 0.25, label = 'Horizontal Average')
    plt.title("Combined Convective Flux Plot for Nominal Ra= "+nomRa)
    plt.legend(loc = 'upper right')
    plt.xlabel('z')
    plt.ylabel("Flux")
    file_name = path+"/"+name+"/"+nomRa+"nominal_CombinedConvective_fig.png"
    plt.savefig(file_name)
    print(file_name)
    plt.close()