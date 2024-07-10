from docopt import docopt
from configparser import ConfigParser
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import mpi4py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import matplotlib.pyplot as plt
import sys
import os 
import csv
path = os.path.dirname(os.path.abspath(__file__))
from scipy.optimize import minimize_scalar
from EVP_methods import geteigenval , modesolver
import time
# configfile = path +"/options.cfg"
# args = docopt(__doc__)
if len(sys.argv) < 2:
    # raise
    try:
        configfile = path + "/options.cfg"
    except:
        print('please provide config file')
        raise
else:
    configfile = sys.argv[1]
#Config file
config = ConfigParser()
config.read(str(configfile))
name = config.get('param', 'name')
#Eigenvalue Spectrum Function
def getgrowthrates(Rayleigh, Prandtl,Nz, A, ad,sig, NEV=10, target=0):
    comm = MPI.COMM_WORLD
    # Compute growth rate over local wavenumbers
    kx_local = kx_global[comm.rank::comm.size]
    t1 = time.time()
    # for all 
    growth_locallist = []
    frequecny_locallist = []
    for kx in kx_local:
        eigenvals = geteigenval(Rayleigh, Prandtl, kx, Nz,A,ad,sig, NEV=NEV) #np.array of complex
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
    ratelist=[]
    comm.barrier()
    
    comm.Bcast(growth_global,root=0)
    for i in growth_global:
        ratelist.append(i)
    return ratelist

def findmarginalomega(Rayleigh, Prandtl, Nz, A, ad,sig):
    counter = 0
    growthrateslist=getgrowthrates(Rayleigh, Prandtl, Nz, A, ad,sig, NEV=10, target = 0)
    if rank == 0:
        print('Z Resolution:',Nz)
        print('Rayleigh:', Rayleigh)
        print('Sigma:', sig)
        print('Intial Ampllitude:', A)
    if rank == 0:
        print('Intial Growth Rates:',growthrateslist)
    max_omeg = max(growthrateslist)
    if rank == 0:
        print('intial parameters maximum growth rate',max_omeg)
    #Finding marginal stability
    A_plus = A+epsilon
    A_minus= A-epsilon
    plusamp_list=getgrowthrates(Rayleigh, Prandtl, Nz, A_plus, ad,sig, NEV=10, target = 0)
    ampomeg_plus=max(plusamp_list) 
    minusamp_list=getgrowthrates(Rayleigh, Prandtl, Nz, A_minus, ad,sig, NEV=10, target = 0)
    ampomeg_minus=max(minusamp_list)
    omeg_guess = np.inf

    #Plotting test amplitudes
    doamptest = config.getboolean('param', 'plotornot')
    if doamptest:
        Amp_list = np.linspace(2,4,20)
        guessrates_solve = []
        for i in Amp_list:
            guessrates_solve.append(max(getgrowthrates(Rayleigh, Prandtl, Nz, i, ad,sig)))
        if rank == 0: 
            print(guessrates_solve)
        if rank == 0: 
            # print(guessrates_solve)
            plt.scatter(Amp_list,guessrates_solve)
            plt.xlabel('Amplitudes')
            plt.ylabel(r'Growth Guess ($\omega_{guess}$)')
            plt.title('Ra={}'.format(Rayleigh)+' Sig={}'.format(sig)+' Nz='+str(Nz))
            full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'Ra{}'.format(Rayleigh)+'Pr{}'.format(Prandtl)+'/sig{}/'.format(sig)+'/'
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)
            plt.savefig(full_dir+'Ra={}'.format(Rayleigh)+'Sig={}'.format(sig)+'Nz={}'.format(Nz)+'_amplitudesvsomeg_guess.png')
    else:
    # Rootsolving
        while abs(0-omeg_guess) > tol:
            print('rank={}'.format(rank))
            ispluscloser = abs(ampomeg_plus) < abs(ampomeg_minus)
            A_guess = (A_plus*(ampomeg_minus)-A_minus*(ampomeg_plus))/(ampomeg_minus-ampomeg_plus)
            finalrates = getgrowthrates(Rayleigh, Prandtl, Nz, A_guess, ad,sig, NEV=10, target = 0)
            omeg_guess = max(finalrates)
            if ispluscloser: 
                A_minus = A_guess 
                A = A_guess
                ampomeg_minus = omeg_guess
            else:
                A_plus = A_guess
                ampomeg_plus = omeg_guess
                A = A_guess
            counter = counter + 1
            if rank == 0:
                print("ampomeg_plus={}".format(ampomeg_plus))
                print("ampomeg_minus={}".format(ampomeg_minus))
                print("omeg_guess={}".format(omeg_guess))
                print("A={}".format(A_guess))
                print("tol={}".format(tol))
                print('\n\n'+'Iteration #:', str(counter)+'\n\n' )
        if abs(0-omeg_guess) < tol: 
            #Printing final rates
            if rank == 0:
                print(finalrates) #Growth rates list containing marginally stable mode
                print(wavenum_list) #List of wavenumbers
            #Finding the dominate mode wavenumber (kx)
            for i in range(len(finalrates)):
                omega_final = finalrates[i]
                if omega_final == omeg_guess:
                    maxomeg_kx = wavenum_list[i]
            #Writing conditions for marginal stability -> IN .CSV FILE
            if rank == 0:
                full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'Ra{}'.format(Rayleigh)+'Pr{}'.format(Prandtl)+'/sig{}/'.format(sig)+'/'
                if not os.path.exists(full_dir):
                    os.makedirs(full_dir)
                csvname = full_dir+'Nz{}'.format(Nz)+'.csv'
                with open(csvname, 'w', newline='') as csvfile:
                    stabilitylog = csv.writer(csvfile, delimiter=',',
                                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                    stabilitylog.writerow("Condtions for marginal stability:"+'\n'+'|------------------|'+'\n')
                    stabilitylog.writerow('Z Resolution: '+str(Nz))
                    stabilitylog.writerow('Tolerance: '+str(tol))
                    stabilitylog.writerow('Rayleigh Number: '+str(Rayleigh))
                    stabilitylog.writerow('Prandtl Number: '+str(Prandtl))
                    stabilitylog.writerow('(mean) Adiabat: '+str(ad))
                    stabilitylog.writerow('Amplitude: '+str(A))
                    stabilitylog.writerow('Wavenumber (kx) for maximum growth rate: '+str(maxomeg_kx))
                    stabilitylog.writerow('Maximum growth rate: '+str(omeg_guess))
            #Writing conditions for marginal stability -> IN TERMINAL
            if rank == 0:
                print("###################################################################################")
                print("###################################################################################")
                print("Condtions for marginal stability:")
                print('Rayleigh Number:', Rayleigh)
                print('Prandtl Number:', Prandtl)
                print('Adiabat:', ad)
                print('Amplitude:', A)
                print('Wavenumber (kx) for maximum growth rate:', maxomeg_kx)
                print("###################################################################################")
                print("###################################################################################")
            results = [Rayleigh, sig,maxomeg_kx, A]
    return 

def modewrapper(Rayleigh, Prandtl, kx, Nz, A_ad, adiabat_mean, sig,Lz,NEV=10, target=0):
    
    print('Mode conditions:\n\n')
    print('Rayleigh:', Rayleigh)
    print('Sig:', sig)
    #Solver evaluated
    solver = modesolver(Rayleigh, Prandtl, kx, Nz, A_ad, adiabat_mean, sig,Lz,NEV=10, target=0)
    sp = solver.subproblems[0]
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    evals = evals[np.argsort(-evals.real)]
    print(f"Slowest decaying mode: λ = {evals[0]}")
    solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0])
    #Buoyancy field
    b = solver.state[1]
    b.change_scales(1)
    #Phases
    pi=np.pi
    phase=0
    phaser=np.exp(((1j*phase)*(2*pi))/4)
    #Modes
    arr_x = np.linspace(0,4,256)
    mode=np.exp(1j*kx*arr_x)
    b_mode=(np.outer(b['g'],mode)*phaser).real
    return b_mode

# Parameters
Nz = config.getint('param', 'Nz')
Nx = config.getint('param','Nx')
Rayleigh = config.getfloat('param', 'Ra') 
Prandtl = config.getfloat('param', 'Pr')
Lz = config.getfloat('param','Lz')
L_x = config.getfloat('param','Lx')
pi=np.pi
kx_global =eval(config.get('param','kx_global'))
wavenum_list = []
for i in kx_global:
    wavenum_list.append(i)
maxomeg_kx = 0
if rank == 0:
    print('Wavenumbers :',wavenum_list)
NEV = 1
A = config.getfloat('param','A')
sig = sig_og = config.getfloat('param','sig')
ad = config.getfloat('param', 'adiabat_mean')  
#Search parameters
epsilon = config.getfloat('param','epsilon')
tol = config.getfloat('param','tol')

#Plotting

# Bases
zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord, dtype=np.complex128, comm=MPI.COMM_SELF)
zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, 1))
z = dist.local_grid(zbasis)
arr_x = np.linspace(0,4,Nx)
aspectratio = 0
fig, ([ax1, ax2,],[ax3,ax4]) = plt.subplots(2, 2)
fig.tight_layout()

#Top left corner
ax1.set_aspect('equal')
ax1.set_adjustable('box', share=True)
ax1.set_ylabel('Ra='+str(Rayleigh))
# margsoln=findmarginalomega(Rayleigh, Prandtl, Nz, A, ad,sig)
#Mode paramaters
if rank == 0:
    A = 1.27763189473734 #margsoln[3]
    kx = 3.141592653589793#margsoln[2]
    soln = modewrapper(Rayleigh, Prandtl, kx, Nz, A, ad, sig,Lz,NEV=10, target=0)
    c = ax1.pcolormesh(arr_x,z,soln-2*(z[..., np.newaxis]-1/2), cmap='RdBu') 
    fig.colorbar(c, ax=ax1)

# #Top right corner
# ax = axs[0, 1]
Rayleigh=1e3
sig=0.003
ax2.set_aspect('equal')
ax2.set_adjustable('box', share=True)
# margsoln=findmarginalomega(Rayleigh, Prandtl, Nz, A, ad,sig)
#Mode paramaters
A = 1.186577903268805#margsoln[3]
kx =3.141592653589793 #margsoln[2]
if rank == 0:
    soln = modewrapper(Rayleigh, Prandtl, Nz, ad, sig, A, kx)
c = ax2.pcolormesh(arr_x,z,soln,cmap='RdBu')
fig.colorbar(c, ax=ax2)

#Bottom left corner
# ax = axs[1, 0]
Rayleigh=10
# sig=sig_og
ax3.set_xlabel(r'$\sigma$='+str(sig))
ax3.set_aspect('equal')
ax3.set_adjustable('box', share=True)
ax3.set_ylabel('Ra='+str(Rayleigh))
# margsoln=findmarginalomega(Rayleigh, Prandtl, Nz, A, ad,sig)
#Mode paramaters
A = 74.39846624838417#margsoln[3]
kx = 3.141592653589793#margsoln[2]
if rank == 0:
    soln = modewrapper(Rayleigh, Prandtl, Nz, ad, sig, A, kx)
c = ax3.pcolormesh(arr_x,z,soln,cmap='RdBu')
fig.colorbar(c, ax=ax3)

#Bottom right corner
# ax = axs[1, 1]
Rayleigh=10
sig=0.003
ax4.set_aspect('equal')
ax4.set_adjustable('box', share=True)
ax4.set_xlabel(r'$\sigma$='+str(sig))
# margsoln=findmarginalomega(Rayleigh, Prandtl, Nz, A, ad,sig)
#Mode paramaters
A = 75.69009237957283#margsoln[3]
kx = 3.141592653589793#margsoln[2]
if rank == 0:
    soln = modewrapper(Rayleigh, Prandtl, Nz, ad, sig, A, kx)
c = ax4.pcolormesh(arr_x,z,soln,cmap='RdBu')
fig.colorbar(c, ax=ax4)

if rank == 0:
    ra1 = 1e3
    ra2 = 10
    sig1 = 0.008
    sig2 = 0.003
    full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'Ra1{}'.format(ra1)+'Ra2{}'.format(ra2)+'_modeplots/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    plt.savefig(full_dir+'Sig1={}'.format(sig1)+'Sig2={}'.format(sig2)+'_multimode.png')
    # folderstring= "Ra"+str(Rayleigh)+"Pr"+str(Prandtl)
    if rank == 0:
        print(path+"/"+name+"multipanelheatmode.png")
    plt.close()