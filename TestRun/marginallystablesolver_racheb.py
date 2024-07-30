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
from scipy.optimize import minimize_scalar, root_scalar
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
target = config.getfloat('param','target')
sig = sig_og = config.getfloat('param','sig')
ad = config.getfloat('param','back_ad')
#Search parameters
epsilon = config.getfloat('param','epsilon')
tol = config.getfloat('param','tol')
name = config.get('param', 'name')
#Single core printing
def print_rank(string):
    if rank == 0:
        print(string)
    return
#Eigenvalue Spectrum Function
def getgrowthrates(Rayleigh, Prandtl,Nz, ad, sig,Lz):
    comm = MPI.COMM_WORLD
    # Compute growth rate over local wavenumbers
    kx_local = kx_global[comm.rank::comm.size]
    if rank == 0:
        print(kx_local)
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
        freq = eigenvals[max_index].real
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
    ratelist,freqlist=[],[]
    comm.barrier()
    
    comm.Bcast(growth_global,root=0)
    for i in growth_global:
        ratelist.append(i)
    for i in freq_global:
        freqlist.append(i)
    return [ratelist,freqlist]

def findmarginalomega(Rayleigh, Prandtl,Nz, ad, sig,Lz):
    counter = 0
    growthrateslist=getgrowthrates(Rayleigh, Prandtl,Nz, ad, sig,Lz)[0]
    max_omeg = max(growthrateslist)
    if rank == 0:
        print('Z Resolution:',Nz)
        print('Intial Rayleigh:', Rayleigh)
        print('Sigma:', sig)
        print('Background adiabat:', ad)
        print('#############')
        print('Intial parameters maximum growth rate',max_omeg)
        print('#############')
        print('Intial Growth Rates:',growthrateslist)
    # Finding marginal stability
    ra_mean = 0
    mean_eig = 0
    margstabilitycriterion = np.abs(max_omeg-2*tol) < tol
    ratelist = []
    if margstabilitycriterion:
        ra_mean = Rayleigh
        mean_eig = max_omeg
        ratelist = growthrateslist
        margconvergence = True
    else:
        margconvergence = False
        if max_omeg > 3*tol:
            ra_plus = Rayleigh
            ra_minus = ra_plus
            minusfound = False
            while not minusfound:
                ra_minus = ra_minus/epsilon
                if rank == 0: 
                    print('Rayleigh:',ra_minus)
                ratelist = getgrowthrates(ra_minus, Prandtl,Nz, ad, sig,Lz)[0]
                minus_eig = max(ratelist)
                if rank == 0:
                    print(plus_eig)
                minusfound = minus_eig < tol/2
        if max_omeg < tol:
            ra_minus = Rayleigh
            ra_plus = ra_minus
            plusfound = False
            while not plusfound:
                ra_plus = ra_plus*epsilon
                if rank == 0: 
                    print('Rayleigh:',ra_plus)
                ratelist = getgrowthrates(ra_plus, Prandtl,Nz, ad, sig,Lz)[0]
                plus_eig = max(ratelist)
                if rank == 0:
                    print(plus_eig)
                plusfound = plus_eig > tol*3
        countercriterion = counter <= 10
        while (not margconvergence) & countercriterion:
            ra_mean = (ra_minus+ra_plus)/2
            ratelist = getgrowthrates(ra_mean, Prandtl,Nz, ad, sig,Lz)[0]
            mean_eig = max(ratelist)
            counter = counter + 1 
            if rank == 0:
                print('Iteration: ', counter)
                print('Max growth rate: ',mean_eig)
                print('Rayleigh #: ',ra_mean)
            if np.abs(mean_eig-2*tol) < tol:
                margconvergence = True
            elif mean_eig >  3*tol:
                ra_plus = ra_mean
            elif mean_eig < tol:
                ra_minus = ra_mean
    finalrates = ratelist
    # Found marginal stability
    if margconvergence: 
        #Printing final rates
        if rank == 0:
            print(finalrates) #Growth rates list containing marginally stable mode
            print(wavenum_list) #List of wavenumbers
        #Finding the dominate mode wavenumber (kx)
        for i in range(len(finalrates)):
            omega_final = finalrates[i]
            if omega_final == mean_eig:
                maxomeg_kx = wavenum_list[i]
        #Writing conditions for marginal stability -> IN .CSV FILE
        if rank == 0:
            full_dir = '/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/'+'AD{}'.format(ad)+'sig{}'.format(sig)+'/'
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)
            csvname = full_dir+'Nz{}'.format(Nz)+'kx{}'.format(len(kx_global))+'.csv'
            with open(csvname, 'w', newline='') as csvfile:
                stabilitylog = csv.writer(csvfile, delimiter=',',
                                        quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                stabilitylog.writerow("Condtions for marginal stability:"+'\n'+'|------------------|'+'\n')
                stabilitylog.writerow('Z Resolution: '+str(Nz))
                stabilitylog.writerow('Tolerance: '+str(tol))
                stabilitylog.writerow('Marginal Rayleigh Number: '+str(ra_mean))
                stabilitylog.writerow('Prandtl Number: '+str(Prandtl))
                stabilitylog.writerow('Background Adiabat: '+str(ad))
                stabilitylog.writerow('Sigma: '+str(sig))
                stabilitylog.writerow('Wavenumber (kx) for maximum growth rate: '+str(maxomeg_kx))
                stabilitylog.writerow('Maximum growth rate: '+str(mean_eig))
            full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'AD{}'.format(ad)+'sig{}'.format(sig)+'/'
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)
            csvname = full_dir+'Nz{}'.format(Nz)+'kx{}'.format(len(kx_global))+'.csv'
            with open(csvname, 'w', newline='') as csvfile:
                stabilitylog = csv.writer(csvfile, delimiter=',',
                                        quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                stabilitylog.writerow("Condtions for marginal stability:"+'\n'+'|------------------|'+'\n')
                stabilitylog.writerow('Z Resolution: '+str(Nz))
                stabilitylog.writerow('Tolerance: '+str(tol))
                stabilitylog.writerow('Marginal Rayleigh Number: '+str(ra_mean))
                stabilitylog.writerow('Prandtl Number: '+str(Prandtl))
                stabilitylog.writerow('Background Adiabat: '+str(ad))
                stabilitylog.writerow('Sigma: '+str(sig))
                stabilitylog.writerow('Wavenumber (kx) for maximum growth rate: '+str(maxomeg_kx))
                stabilitylog.writerow('Maximum growth rate: '+str(mean_eig))
        #Writing conditions for marginal stability -> IN TERMINAL
        if rank == 0:
            print("###################################################################################")
            print("###################################################################################")
            print("Condtions for marginal stability:")
            print('Marginal Rayleigh Number:', ra_mean)
            print('Prandtl Number:', Prandtl)
            print('Background Adiabat:', ad)
            print('Sigma: ',sig)
            print('Wavenumber (kx) for maximum growth rate:', maxomeg_kx)
            print("###################################################################################")
            print("###################################################################################")
        results = [ra_mean, sig,maxomeg_kx, ad]
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

def growthratescurve(ra_list,Prandtl,Nz, ad, sig,Lz):
    if rank == 0: 
        print('\n')
        print('###########')
        print('Conditions')
        print('Lowest Ra: ',str(min(ra_list)))
        print('Upper Ra: ',str(max(ra_list)))
        print('Pr: ',str(Prandtl))
        print('Sigma: ',str(sig))
        print('Adiabat: ',str(ad))
        print('Z Resolution: ',str(Nz))
        print('###########')
        print('\n')
    guessrates_solve = []
    nKx = len(kx_global)
    nRa = len(ra_list)
    growth_mat, freq_mat= np.zeros((nKx,nRa)),np.zeros((nKx,nRa))
    for index, ra in enumerate(ra_list):
        eigenvals_list = getgrowthrates(ra, Prandtl,Nz, ad, sig,Lz)
        ratelist = eigenvals_list[0]
        freqlist = eigenvals_list[1]
        if rank == 0:
            print(freqlist)
        maxomeg = max(ratelist)
        growth_mat[:,index]=np.array(ratelist).T
        freq_mat[:,index]=np.array(freqlist).T
        #     print('Frequencies: ',freq_mat)
        if rank == 0:
            print('Kx: ',kx_global[np.argmax(ratelist)])
            print('Kx index: ', np.argmax(ratelist))
            print('Index=',str(index+1))
            runsleft = (len(ra_list)+1)-(index+1)
            print('Runs left',runsleft)
            print('Rayleigh: ',str(ra_list[index]))
            print('Maximum eigenval:',maxomeg)
        guessrates_solve.append(maxomeg)
    if rank == 0:
        # print('growth_mat:',growth_mat)
        print('freq_mat:',freq_mat)
        print('#########')
        print('Eigenvals:\n',guessrates_solve)
        print('#########')
    from random import randint
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    color = [r, g, b]
    fig, (growth_ax,freq_ax) = plt.subplots(1, 2,sharex='col',sharey='row')
    fig.suptitle(r'Heat map '+r'$\sigma$'+'={} '.format(sig)+r'$\nabla$'+'={} '.format(ad))
    fig.supxlabel(r'Ra')
    fig.supylabel(r'$k_{x}$')
    #Growth rates heat map
    growth_c = growth_ax.pcolormesh(ra_list,kx_global,growth_mat)
    fig.colorbar(growth_c, ax=growth_ax)
    growth_ax.set_title(r'Im($\omega$)',fontsize='small')
    growth_ax.set_xlim(min(ra_list),max(ra_list))
    growth_ax.set_xscale('log')
    #Frequencies heat map
    freq_c = freq_ax.pcolormesh(ra_list,kx_global,freq_mat, cmap='plasma')
    fig.colorbar(freq_c, ax=freq_ax)
    freq_ax.set_title(r'Re($\omega$)',fontsize='small')
    freq_ax.set_xlim(min(ra_list),max(ra_list))
    freq_ax.set_xscale('log')
    return
 
bound_upper=20
bound_lower=4
step_factor=2
powers = np.linspace(bound_lower,bound_upper,step_factor*abs(bound_upper-bound_lower)+1)
testlist = []

for power in powers:
    testlist.append(10**power)
growthratescurve(testlist,Prandtl,Nz,ad,sig,Lz)
full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'ad{}'.format(ad)+'/'
if not os.path.exists(full_dir):
    os.makedirs(full_dir)
bckup_dir = '/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/'+'ad{}'.format(ad)+'/'
if not os.path.exists(bckup_dir):
    os.makedirs(bckup_dir)
# plt.savefig(bckup_dir+'ad{}'.format(ad)+'sig{}'.format(sig)+'Nz{}'.format(Nz)+'kx{}'.format(len(kx_global))+'_ranumsvsmean_eig.png') 
# print_rank(bckup_dir+'ad{}'.format(ad)+'sig{}'.format(sig)+'Nz{}'.format(Nz)+'kx{}'.format(len(kx_global))+'_ranumsvsmean_eig.png')
# plt.savefig(full_dir+'ad{}'.format(ad)+'sig{}'.format(sig)+'Nz{}'.format(Nz)+'kx{}'.format(len(kx_global))+'_ranumsvsmean_eig.png')
plt.savefig(path+'/'+name+'/'+name+'_testrunfig.png')
plt.close()
sys.exit()

# sig_list = [0.01]
# for sig in sig_list:
# if rank == 0:
#     print('Sigma: ',sig_list)


# nz_lower=6
# step_nz=1
# nz_powers = np.linspace(nz_lower,nz_upper,step_nz*abs(nz_upper-nz_lower)+1)
# nzlist=[]
# for power in nz_powers:
#     nzlist.append(2**power)
# for Nz in nzlist:
# if rank == 0:
#     print('Running sigma: ',sig)

#Plotting
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
