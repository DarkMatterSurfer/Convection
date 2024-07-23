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
# size = comm.Get_size()
import matplotlib.pyplot as plt
import sys
import os 
import csv
path = os.path.dirname(os.path.abspath(__file__))
from scipy.optimize import minimize_scalar
from EVP_methods_CHEBBED import modesolver,adiabatresolutionchecker
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
# Parameters
Nz = config.getint('param', 'Nz')
Nx = config.getint('param','Nx')
Rayleigh = config.getfloat('param', 'Ra') 
Prandtl = config.getfloat('param', 'Pr')
Re_arg = config.getfloat('param','Re')
Lz = config.getfloat('param','Lz')
Lx = config.getfloat('param','Lx')
pi=np.pi
kx_global =eval(config.get('param','kx_global'))
wavenum_list = []
for i in kx_global:
    wavenum_list.append(i)
maxomeg_kx = 0
if rank == 0:
    print('Wavenumbers :',wavenum_list)
NEV = config.getint('param','NEV')
target = config.getint('param','target')
sig = sig_og = config.getfloat('param','sig')
ad = config.getfloat('param','back_ad')
#Search parameters
epsilon = config.getfloat('param','epsilon')
tol = config.getfloat('param','tol')
name = config.get('param', 'name')
#Eigenvalue Spectrum Function
def getgrowthrates(Rayleigh, Prandtl,Nz, ad, sig,Lz):
    comm = MPI.COMM_WORLD
    # Compute growth rate over local wavenumbers
    kx_local = kx_global[comm.rank::comm.size]
    t1 = time.time()
    # for all 
    growth_locallist = []
    frequecny_locallist = []
    # if rank == 0:
    #     print('here')
    for kx in kx_local:
        if rank == 0:
            print('2 here. In getgrowthrates')
        eigenvals = modesolver(Rayleigh, Prandtl, kx, Nz, ad, sig,Lz,NEV, target).eigenvalues #np.array of complex
        eigenlen = len(eigenvals)
        gr_max = -1*np.inf
        max_index = -1
        for i in range(eigenlen):
            if (eigenvals[i].imag) > gr_max:
                gr_max=(eigenvals[i].imag)
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

def findmarginalomega(Rayleigh, Prandtl,Nz, ad, sig,Lz):
    counter = 0
    # if rank ==0:
    #     print('here')
    if rank == 0:
        print('1 here. In findmarginalomega/growthrateslist')
    growthrateslist=getgrowthrates(Rayleigh, Prandtl,Nz, ad, sig,Lz)
    max_omeg = max(growthrateslist)
    if rank == 0:
        print('Z Resolution:',Nz)
        print('Intial Rayleigh:', Rayleigh)
        print('Sigma:', sig)
        print('Background adiabat:', ad)
        print('#############')
        print('Intial parameters maximum growth rate',max_omeg)
        print('#############')
    if rank == 0:
        print('Intial Growth Rates:',growthrateslist)
    #Finding marginal stability
    Ra_plus = Rayleigh*epsilon
    Ra_minus= Rayleigh/epsilon
    plusamp_list=getgrowthrates(Ra_plus, Prandtl,Nz, ad, sig,Lz)
    omeg_plusRa=max(plusamp_list) 
    minusamp_list=getgrowthrates(Ra_minus, Prandtl,Nz, ad, sig,Lz)
    omeg_minusRa=max(minusamp_list)
    omeg_guess = np.inf

    #Plotting test amplitudes
    doamptest = config.getboolean('param', 'plotornot')
    if doamptest:
        ra_list = 10**np.linspace(1,15,20)
        # ra_list = np.linspace(1e3,1e6,40)
        guessrates_solve = []
        if rank == 0:
            print('here. In findmarginalomega/guessratessolve loop')
        for index, i in enumerate(ra_list):
            if rank == 0:
                print('Index=',str(index))
                print('Runs left',str(len(ra_list+1)-(index+1)))
                print('Rayleigh: ',str(ra_list[index]))
            guessrates_solve.append(max(getgrowthrates(ra_list[index], Prandtl,Nz, ad, sig,Lz)))
        if rank == 0: 
            print(guessrates_solve)
        if rank == 0:
            print('here. In findmarginalomega/figureplotting')
        if rank == 0: 
            # print(guessrates_solve)
            plt.scatter(ra_list,guessrates_solve)
            plt.xscale('log')
            plt.xlabel('Rayleigh Number')
            plt.ylabel(r'Growth Guess ($\omega_{guess}$)')
            plt.title(r'$\nabla_{ad}$='+'{}'.format(ad)+' Sig={}'.format(sig)+' Nz='+str(Nz))
            full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'sig{}'.format(sig)+'/'+'AD{}'.format(ad)+'/'
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)
            plt.savefig(full_dir+'AD={}'.format(ad)+'Nz={}'.format(Nz)+'_ranumsvsomeg_guess.png')
    else:
    # Rootsolving
        while abs(0-omeg_guess) > tol:
            ispluscloser = abs(omeg_plusRa) < abs(omeg_minusRa)
            ra_guess = (Ra_plus*(omeg_minusRa)-Ra_minus*(omeg_plusRa))/(omeg_minusRa-omeg_plusRa)
            if ra_guess < 1:
                print('')
                break
            finalrates = getgrowthrates(ra_guess, Prandtl,Nz, ad, sig,Lz)
            omeg_guess = max(finalrates)
            if ispluscloser: 
                Ra_minus = ra_guess 
                # A = ra_guess
                omeg_minusRa = omeg_guess
            else:
                Ra_plus = ra_guess
                omeg_plusRa = omeg_guess
                # A = ra_guess
            counter = counter + 1
            if rank == 0:
                print("omeg_plusRa={}".format(omeg_plusRa))
                print("omeg_minusRa={}".format(omeg_minusRa))
                print("omeg_guess={}".format(omeg_guess))
                print("Rayliegh={}".format(ra_guess))
                print("tol={}".format(tol))
                print('\n\n'+'Iteration #:', str(counter)+'\n\n' )
        # Found marginal stability
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
                full_dir = '/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/'+'AD{}'.format(ad)+'sig{}'.format(sig)+'/Rayleigh{}/'.format(Rayleigh)+'/'
                if not os.path.exists(full_dir):
                    os.makedirs(full_dir)
                csvname = full_dir+'Nz{}'.format(Nz)+'.csv'
                with open(csvname, 'w', newline='') as csvfile:
                    stabilitylog = csv.writer(csvfile, delimiter=',',
                                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                    stabilitylog.writerow("Condtions for marginal stability:"+'\n'+'|------------------|'+'\n')
                    stabilitylog.writerow('Z Resolution: '+str(Nz))
                    stabilitylog.writerow('Tolerance: '+str(tol))
                    stabilitylog.writerow('Marginal Rayleigh Number: '+str(ra_guess))
                    stabilitylog.writerow('Prandtl Number: '+str(Prandtl))
                    stabilitylog.writerow('Background Adiabat: '+str(ad))
                    stabilitylog.writerow('Strip Adiabat: '+str(1))
                    stabilitylog.writerow('Sigma: '+str(sig))
                    stabilitylog.writerow('Wavenumber (kx) for maximum growth rate: '+str(maxomeg_kx))
                    stabilitylog.writerow('Maximum growth rate: '+str(omeg_guess))
                full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'AD{}'.format(ad)+'sig{}'.format(sig)+'/Rayleigh{}/'.format(Rayleigh)+'/'
                if not os.path.exists(full_dir):
                    os.makedirs(full_dir)
                csvname = full_dir+'Nz{}'.format(Nz)+'.csv'
                with open(csvname, 'w', newline='') as csvfile:
                    stabilitylog = csv.writer(csvfile, delimiter=',',
                                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                    stabilitylog.writerow("Condtions for marginal stability:"+'\n'+'|------------------|'+'\n')
                    stabilitylog.writerow('Z Resolution: '+str(Nz))
                    stabilitylog.writerow('Tolerance: '+str(tol))
                    stabilitylog.writerow('Marginal Rayleigh Number: '+str(ra_guess))
                    stabilitylog.writerow('Prandtl Number: '+str(Prandtl))
                    stabilitylog.writerow('Background Adiabat: '+str(ad))
                    stabilitylog.writerow('Strip Adiabat: '+str(1))
                    stabilitylog.writerow('Sigma: '+str(sig))
                    stabilitylog.writerow('Wavenumber (kx) for maximum growth rate: '+str(maxomeg_kx))
                    stabilitylog.writerow('Maximum growth rate: '+str(omeg_guess))
            #Writing conditions for marginal stability -> IN TERMINAL
            if rank == 0:
                print("###################################################################################")
                print("###################################################################################")
                print("Condtions for marginal stability:")
                print('Marginal Rayleigh Number:', ra_guess)
                print('Prandtl Number:', Prandtl)
                print('Background Adiabat:', ad)
                print('Sigma: ',sig)
                print('Wavenumber (kx) for maximum growth rate:', maxomeg_kx)
                print("###################################################################################")
                print("###################################################################################")
            results = [ra_guess, sig,maxomeg_kx, ad]
            comm.barrier()
    return results

def modewrapper(Rayleigh, Prandtl, kx, Nx,Nz, ad, sig,Lx,Lz,NEV, target):
    print('Mode conditions:\n\n')
    print('Rayleigh:', Rayleigh)
    print('Sig:', sig)
    #Solver evaluated
    solver = modesolver(Rayleigh, Prandtl, kx, Nz, ad, sig,Lz,NEV, target)
    sp = solver.subproblems[0]
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    evals = evals[np.argsort(evals.imag)]
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
    arr_x = np.linspace(0,Lx,Nx)
    mode=np.exp(1j*kx*arr_x)
    b_mode=(np.outer(b['g'],mode)*phaser).real
    return b_mode

def growthratescurve(Prandtl,Nz, ad, sig,Lz):
    ra_list = 10**np.linspace(1,25,30)
    # ra_list = np.linspace(1e3,1e6,40)
    guessrates_solve = []
    if rank == 0:
        print('here. In findmarginalomega/guessratessolve loop')
        for index, i in enumerate(ra_list):
            print('Index=',str(index))
            print('Runs left',str(len(ra_list+1)-(index+1)))
            print('Rayleigh: ',str(ra_list[index]))
            guessrates_solve.append(max(getgrowthrates(ra_list[index], Prandtl,Nz, ad, sig,Lz)))
        print(guessrates_solve)
        print('here. In findmarginalomega/figureplotting')
        # print(guessrates_solve)
        color = list(np.random.choice(range(256), size=3))
        plt.scatter(ra_list,guessrates_solve,label=str(Nz),c=color)
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Rayleigh Number')
        plt.ylabel(r'Growth Guess ($\omega_{guess}$)')
        plt.title(r'$\nabla_{ad}$='+'{}'.format(ad)+' Sig={}'.format(sig))
    return 
powerNz = np.linspace(6,6,1)
listNz = []
for i in range(len(powerNz)):
    listNz.append(2**i)
if rank == 0:
    print(listNz)
for resolutionNz in powerNz:
    growthratescurve(Prandtl,resolutionNz, ad, sig,Lz)
full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'ad{}'.format(ad)+'/'
if not os.path.exists(full_dir):
    os.makedirs(full_dir)
plt.savefig(full_dir+'ad{}'.format(ad)+'sig{}'.format(sig)+'_ranumsvsomeg_guess.png')
plt.close()
sys.exit()

findmarginalomega(Rayleigh, Prandtl,Nz, ad, sig,Lz)
if rank == 0:
    adiabatresolutionchecker(ad,sig,Nz,Lz,path)
#Plotting
ad_list = np.linspace(1,9,9)
if rank == 0:
    print(ad_list)
marginalRalist = []
ra_1 = findmarginalomega(Rayleigh, Prandtl,Nz, ad_list[0], sig,Lz)[0]
marginalRalist.append(ra_1)
ra_2 = findmarginalomega(marginalRalist[0], Prandtl,Nz, ad_list[1], sig,Lz)[0]
marginalRalist.append(ra_2)
ra_3 = findmarginalomega(marginalRalist[1], Prandtl,Nz, ad_list[2], sig,Lz)[0]
marginalRalist.append(ra_3)
ra_4 = findmarginalomega(marginalRalist[2], Prandtl,Nz, ad_list[3], sig,Lz)[0]
marginalRalist.append(ra_4)
ra_5 = findmarginalomega(marginalRalist[3], Prandtl,Nz, ad_list[4], sig,Lz)[0]
marginalRalist.append(ra_5)
ra_6 = findmarginalomega(marginalRalist[4], Prandtl,Nz, ad_list[5], sig,Lz)[0]
marginalRalist.append(ra_6)
ra_7 = findmarginalomega(marginalRalist[5], Prandtl,Nz, ad_list[6], sig,Lz)[0]
marginalRalist.append(ra_7)
ra_8 = findmarginalomega(marginalRalist[6], Prandtl,Nz, ad_list[7], sig,Lz)[0]
marginalRalist.append(ra_8)
ra_9 = findmarginalomega(marginalRalist[7], Prandtl,Nz, ad_list[8], sig,Lz)[0]
marginalRalist.append(ra_9)
print(ad_list)
print(marginalRalist)
findmarginalomega(Rayleigh, Prandtl,Nz, ad, sig,Lz)
raorigin = findmarginalomega(Rayleigh, Prandtl,Nz, ad_list[0], sig,Lz)[0]
marginalRalist.append(raorigin)
for i in range(len(ad_list)):
    #if on second adiabat entry
    if not (ad_list[i] == ad_list[0]):
        print(marginalRalist[i-1])
        margRa = findmarginalomega(marginalRalist[i-1],Prandtl,ad_list[i],Nz,sig,Lz)[0]
        marginalRalist.append(margRa)
# Bases
zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord, dtype=np.complex128, comm=MPI.COMM_SELF)
zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, Lz))
z = dist.local_grid(zbasis)
arr_x = np.linspace(0,Lx,Nx)
aspectratio = 0
fig, ([ax1, ax2,],[ax3,ax4]) = plt.subplots(2, 2)
fig.tight_layout()

#Top left corner
ax1.set_aspect('equal')
ax1.set_adjustable('box', share=True)
ax1.set_ylabel('Ra='+str(Rayleigh))
margsoln=findmarginalomega(Rayleigh, Prandtl, Nx,Nz, ad, sig,Lx,Lz, NEV, target)
#Mode paramaters
if rank == 0:
    A = margsoln[3]
    kx = margsoln[2]
    soln = modewrapper(Rayleigh, Prandtl, kx, Nx,Nz, ad, sig,Lx,Lz,NEV, target)
    c = ax1.pcolormesh(arr_x,z-2*(z[..., np.newaxis]-1/2),soln, cmap='RdBu') 
    fig.colorbar(c, ax=ax1)

#Top right corner
ax = axs[0, 1]
Rayleigh=1e3
sig=0.003
ax2.set_aspect('equal')
ax2.set_adjustable('box', share=True)
margsoln=findmarginalomega(Rayleigh, Prandtl, Nz, A, ad,sig)
#Mode paramaters
A = margsoln[3]
kx =margsoln[2]
if rank == 0:
    soln = modewrapper(Rayleigh, Prandtl, kx, Nz, A, ad, sig,Lz,Nx,NEV=10, target=0) 
c = ax2.pcolormesh(arr_x,z,soln,cmap='RdBu')
fig.colorbar(c, ax=ax2)

# #Bottom left corner
Rayleigh=10
sig=sig_og
ax3.set_xlabel(r'$\sigma$='+str(sig))
ax3.set_aspect('equal')
ax3.set_adjustable('box', share=True)
ax3.set_ylabel('Ra='+str(Rayleigh))
margsoln=findmarginalomega(Rayleigh, Prandtl, Nz, A, ad,sig)
#Mode paramaters
A = margsoln[3]
kx = margsoln[2]
if rank == 0:
    soln = modewrapper(Rayleigh, Prandtl, kx, Nz, A, ad, sig,Lz,Nx,NEV=10, target=0) 
c = ax3.pcolormesh(arr_x,z,soln,cmap='RdBu')
fig.colorbar(c, ax=ax3)

#Bottom right corner
# ax = axs[1, 1]
Rayleigh=10
sig=0.003
ax4.set_aspect('equal')
ax4.set_adjustable('box', share=True)
ax4.set_xlabel(r'$\sigma$='+str(sig))
margsoln=findmarginalomega(Rayleigh, Prandtl, Nz, A, ad,sig)
#Mode paramaters
A = margsoln[3]
kx = margsoln[2]
if rank == 0:
    soln = modewrapper(Rayleigh, Prandtl, kx, Nz, A, ad, sig,Lz,Nx,NEV=10, target=0) 
c = ax4.pcolormesh(arr_x,z,soln,cmap='RdBu')
fig.colorbar(c, ax=ax4)

if rank == 0:
    ra1 = 1e3
    ra2 = 0
    sig1 = 0.01
    sig2 = 0.008
    full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'Ra1{}'.format(ra1)+'Ra2{}'.format(ra2)+'_modeplots/'
    full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    figpath=full_dir+'Sig1={}'.format(sig1)+'Sig2={}'.format(sig2)+'_multimode.png'
    figpath=full_dir+name+'testing_multimode.png'
    plt.savefig(figpath)
    if rank == 0:
        print(figpath)
    plt.close()
