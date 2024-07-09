import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
import sys
import mpi4py
from mpi4py import MPI
from EVP_methods import geteigenval 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
path = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)
if len(sys.argv) < 2:
    # raise
    try:
        configfile = path + "/options.cfg"
    except:
        print('please provide config file')
        raise
else:
    configfile = sys.argv[1]
from configparser import ConfigParser
config = ConfigParser()
config.read(str(configfile))


import time
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD

# Parameters
Nz = config.getfloat('param', 'Nz')
Nx = config.getfloat('param','Nx')
Rayleigh = config.getfloat('param', 'Ra') 
Prandtl = config.getfloat('param', 'Pr')
L_x = config.getfloat('param','Lx')
Lz = config.getfloat('param','Lz')
pi=np.pi
kx_global = eval(config.get('param','kx_global'))
wavenum_list = []
NEV = config.getint('param','NEV')
A = config.getfloat('param','A')
sig = sig_og = config.getfloat('param','sig')
ad = config.getfloat('param', 'adiabat_mean')  
def getgrowthrates(Rayleigh, Prandtl,Nz, A, ad,sig, NEV=NEV, target=0):
    # Compute growth rate over local wavenumbers
    kx_local = kx_global[comm.rank::comm.size]
    t1 = time.time()
    # for all 
    growth_locallist = []
    frequecny_locallist = []
    for kx in kx_local:
        eigenvals = geteigenval(Rayleigh, Prandtl, kx, Nz,A,ad,sig,Lz=Lz, NEV=NEV) #np.array of complex
        eigenlen = len(eigenvals)
        gr_max = -1*np.inf
        max_index = -1
        for i in range(eigenlen):
            if -1*(eigenvals[i].real) > gr_max:
                gr_max=-1*(eigenvals[i].real)
                max_index = i
        eigenvals[max_index].imag
        freq = eigenvals[max_index].imag
        growth_locallist.append(gr_max)
        frequecny_locallist.append(freq)
    #    growth_locallist.append(np.max())
    growth_local = np.array(growth_locallist)
    freq_local = np.array(frequecny_locallist)
    t2 = time.time()
    logger.info('Elapsed solve time: %f' %(t2-t1))

    # Reduce growth rates to root process
    growth_global = np.zeros_like(kx_global)
    growth_global[comm.rank::comm.size] = growth_local
    if comm.rank == 0:
        comm.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
    else:
        comm.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

    freq_global = np.zeros_like(kx_global)
    freq_global[comm.rank::comm.size] = freq_local
    if comm.rank == 0:
        comm.Reduce(MPI.IN_PLACE, freq_global, op=MPI.SUM, root=0)
    else:
        comm.Reduce(freq_global, freq_global, op=MPI.SUM, root=0)
    results=(growth_global,freq_global)
    return results
# Plot growth rates from root process
growth_global, freq_global = getgrowthrates(Rayleigh, Prandtl,Nz, A, ad,sig, NEV=NEV, target=0)
if comm.rank == 0:
    #Plotting Set-up
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,9), sharex=True, dpi = 500)
    fig.suptitle('Rayleigh Benard Eigenvalue Problem')
    ax1.set_ylabel(r'$\omega$')
    ax2.set_ylabel(r'$\text{f}$')
    # ax1.set_ylim(bottom=0)
    ax2.set_xlabel(r'$k_x$')
    ax1.title.set_text(r'Rayleigh-Benard Modes Growth Rates ($\mathrm{Ra} = %.2f, \; \mathrm{Pr} = %.2f, \; \mathrm{\nabla_{ad}} = %.2f, \; \mathrm{A} = %.2f $)' %(Rayleigh, Prandtl,ad,A))
    ax2.title.set_text(r'Rayleigh-Benard Modes Frequency($\mathrm{Ra} = %.2f, \; \mathrm{Pr} = %.2f, \; \mathrm{\nabla_{ad}} = %.2f, \; \mathrm{A} = %.2f $)' %(Rayleigh, Prandtl,ad,A))

    #Growth Rates
    ax1.scatter(kx_global, growth_global)
    plt.tight_layout()

    #Mode frequency
    ax2.scatter(kx_global, freq_global)
    plt.tight_layout()

    #Figure Saving
    filename = 'Ad_'+str(ad)+'Amp_'+str(A)+'_eigenval_plot.png'
    plt.savefig("/home/Convection/eigenvalprob_plots/"+"Ra"+str(Rayleigh)+"Pr"+str(Prandtl)+"/"+filename)
    print(filename)
    plt.close()