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
from scipy.optimize import minimize_scalar, root_scalar
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
sig = sig_og = config.getfloat('param','sig')
Rayleigh_sig = config.getfloat('param','Ra_sig')
if not Rayleigh_sig == 0:
    Rayleigh = (Rayleigh_sig)/((2*sig)**3)
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
ad = config.getfloat('param','back_ad')
#Search parameters
epsilon = config.getfloat('param','epsilon')
tol = config.getfloat('param','tol')
name = config.get('param', 'name')

def modesolver (Rayleigh, Prandtl, kx, Nz, ad, sig,Lz):

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
    comm.barrier()
    try: 
        zbasis_r  =  d3.ChebyshevT(coords['z'], size=round(Nz/2), bounds=(Lz/2, Lz), dealias=3/2)
        zbasis_c =  d3.ChebyshevT(coords['z'], size=round(Nz/2), bounds=(0, Lz/2), dealias=3/2)
    except Exception as e:
        print(e)
    # Fields
    omega = dist.Field(name='omega')
    tau_p = dist.Field(name='tau_p')
    p_r  = dist.Field(name='p_r', bases=(zbasis_r,))
    T_r  = dist.Field(name='T_r', bases=(zbasis_r,))
    T_z_r  = dist.Field(name='T_z_r', bases=(zbasis_r,))
    ux_r  = dist.Field(name='ux_r', bases=(zbasis_r,))
    uz_r  = dist.Field(name='uz_r', bases=(zbasis_r,))
    ux_z_r  = dist.Field(name='ux_z_r', bases=(zbasis_r,))
    uz_z_r  = dist.Field(name='uz_z_r', bases=(zbasis_r,))
    nabad_r = dist.Field(name='nabad_r', bases=(zbasis_r, ))
    tau_T1_r = dist.Field(name='tau_b1_r')
    tau_T2_r = dist.Field(name='tau_b2_r')
    tau_ux1_r = dist.Field(name='tau_ux1_r')
    tau_ux2_r = dist.Field(name='tau_ux2_r')
    tau_uz1_r = dist.Field(name='tau_uz1_r')
    tau_uz2_r = dist.Field(name='tau_uz2_r')

    p_c  = dist.Field(name='p_c', bases=(zbasis_c,))
    T_c  = dist.Field(name='T_c', bases=(zbasis_c,))
    T_z_c  = dist.Field(name='T_z_c', bases=(zbasis_c,))
    ux_c  = dist.Field(name='ux_c', bases=(zbasis_c,))
    uz_c  = dist.Field(name='uz_c', bases=(zbasis_c,))
    ux_z_c  = dist.Field(name='ux_z_c', bases=(zbasis_c,))
    uz_z_c  = dist.Field(name='uz_z_c', bases=(zbasis_c,))
    nabad_c = dist.Field(name='nabad_c', bases=(zbasis_c, ))
    tau_T1_c = dist.Field(name='tau_b1_c')
    tau_T2_c = dist.Field(name='tau_b2_c')
    tau_ux1_c = dist.Field(name='tau_ux1_c')
    tau_ux2_c = dist.Field(name='tau_ux2_c')
    tau_uz1_c = dist.Field(name='tau_uz1_c')
    tau_uz2_c = dist.Field(name='tau_uz2_c')

    z_r  = dist.local_grids(zbasis_r, )[0]
    z_c = dist.local_grids(zbasis_c, )[0]
    ex, ez = coords.unit_vector_fields(dist)
    z_match = Lz/2
    dz = lambda A: d3.Differentiate(A, coords['z'])
    #Substitutions 
    nu = (Rayleigh / Prandtl)**(-1/2)
    kappa = (Rayleigh*Prandtl)**(-1/2)
    lift_basis_r  = zbasis_r.derivative_basis(1)
    lift_basis_c = zbasis_c.derivative_basis(1)
    lift_r  = lambda A: d3.Lift(A, lift_basis_r, -1)
    lift_c = lambda A: d3.Lift(A, lift_basis_c, -1)
    dt = lambda A: -1j*omega*A
    dx = lambda A: 1j*kx*A
    #Adiabatic Parameterization
    ad_r = ad-(ad-1)*np.exp(-(z_r-(Lz/2))**2*(1/(2*sig**2)))
    ad_c = ad-(ad-1)*np.exp(-(z_c-(Lz/2))**2*(1/(2*sig**2)))
    nabad_r['g']=ad_r
    nabad_c['g']=ad_c

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    vars_r = [p_r, T_r, ux_r, uz_r, T_z_r, ux_z_r, uz_z_r, tau_T1_r, tau_T2_r, tau_ux1_r, tau_uz1_r, tau_ux2_r, tau_uz2_r,tau_p]
    vars_c = [p_c, T_c, ux_c, uz_c, T_z_c, ux_z_c, uz_z_c, tau_T1_c, tau_T2_c, tau_ux1_c, tau_uz1_c, tau_ux2_c, tau_uz2_c]
    problem = d3.EVP(vars_r+vars_c, namespace=locals(), eigenvalue=omega)
    #Top Half
    problem.add_equation("dx(ux_r) + uz_z_r + tau_p = 0")
    problem.add_equation("dt(T_r) - kappa*( dx(dx(T_r)) + dz(T_z_r) ) + lift_r(tau_T2_r) - (-nabad_r+2)*uz_r= 0")
    problem.add_equation("dt(ux_r) - nu*( dx(dx(ux_r)) + dz(ux_z_r) ) + dx(p_r)     + lift_r(tau_ux2_r)= 0")
    problem.add_equation("dt(uz_r) - nu*( dx(dx(uz_r)) + dz(uz_z_r) ) + dz(p_r) - T_r + lift_r(tau_uz2_r) = 0")
    problem.add_equation("T_z_r - dz(T_r) + lift_r(tau_T1_r) = 0")
    problem.add_equation("ux_z_r - dz(ux_r) + lift_r(tau_ux1_r) = 0")
    problem.add_equation("uz_z_r - dz(uz_r) + lift_r(tau_uz1_r) = 0")
    #Bottom Half
    problem.add_equation("dx(ux_c) + uz_z_c = 0")
    problem.add_equation("dt(T_c) - kappa*( dx(dx(T_c)) + dz(T_z_c) ) + lift_c(tau_T2_c) - (-nabad_c+2)*uz_c= 0")
    problem.add_equation("dt(ux_c) - nu*( dx(dx(ux_c)) + dz(ux_z_c) ) + dx(p_c)     + lift_c(tau_ux2_c)= 0")
    problem.add_equation("dt(uz_c) - nu*( dx(dx(uz_c)) + dz(uz_z_c) ) + dz(p_c) - T_c + lift_c(tau_uz2_c) = 0")
    problem.add_equation("T_z_c - dz(T_c) + lift_c(tau_T1_c) = 0")
    problem.add_equation("ux_z_c - dz(ux_c) + lift_c(tau_ux1_c) = 0")
    problem.add_equation("uz_z_c - dz(uz_c) + lift_c(tau_uz1_c) = 0")
    #Matching Conditions
    problem.add_equation("p_r(z=z_match) - p_c(z=z_match) = 0")
    problem.add_equation("ux_r(z=z_match) - ux_c(z=z_match) = 0")
    problem.add_equation("uz_r(z=z_match) - uz_c(z=z_match) = 0")
    problem.add_equation("ux_z_r(z=z_match) - ux_z_c(z=z_match) = 0")
    problem.add_equation("T_r(z=z_match) - T_c(z=z_match) = 0")
    problem.add_equation("T_z_r(z=z_match) - T_z_c(z=z_match) = 0")
    #Boundary Conditions
    problem.add_equation("T_c(z=0) = 0")
    problem.add_equation("ux_c(z=0) = 0")
    problem.add_equation("uz_c(z=0) = 0")
    problem.add_equation("T_r(z=Lz) = 0")
    problem.add_equation("ux_r(z=Lz) = 0")
    problem.add_equation("uz_r(z=Lz) = 0")
    problem.add_equation("integ(p_r) = 0") # Pressure gauge

    # Solver
    
    solver = problem.build_solver()
    sp = solver.subproblems[0]
    try:
        if rank == 0:
            print('here. trying sparse')
        solver.solve_sparse(sp,NEV,target=target,raise_on_mismatch=True)
    except:
        if rank == 0:
            print('sparse solve failed task trying dense solve')
        solver.solve_dense(sp)
    
    # print('rank', str(rank))
    return solver
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
    if rank == 0:
        print('in getgrowthrates')
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
                ratelist = getgrowthrates(ra_minus, Prandtl,Nz, ad, sig,Lz)
                minus_eig = max(ratelist)
                minusfound = minus_eig < tol/2
        if max_omeg < tol:
            ra_minus = Rayleigh
            ra_plus = ra_minus
            plusfound = False
            while not plusfound:
                ra_plus = ra_plus*epsilon
                ratelist = getgrowthrates(ra_plus, Prandtl,Nz, ad, sig,Lz)
                plus_eig = max(ratelist)
                plusfound = plus_eig > tol*3
        countercriterion = counter <= 10
        while (not margconvergence) & countercriterion:
            ra_mean = (ra_minus+ra_plus)/2
            ratelist = getgrowthrates(ra_mean, Prandtl,Nz, ad, sig,Lz)
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
            csvname = full_dir+'Nz{}'.format(Nz)+'.csv'
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
            csvname = full_dir+'Nz{}'.format(Nz)+'.csv'
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

ad_upper=50
ad_lower=1
step_factor=1
ad_list = np.linspace(ad_lower,ad_upper,step_factor*abs(ad_upper-ad_lower)+1)
sig_list=[0.1]
marginalRa = []
raorigin = findmarginalomega(Rayleigh, Prandtl,Nz, ad_list[0], sig,Lz)[0]
marginalRa.append(raorigin)
for i in range(len(ad_list)):
    if not (ad_list[i] == ad_list[0]):
        margsolve = findmarginalomega(marginalRa[i-1],Prandtl,ad_list[0],Nz,sig,Lz)
        margRa = margsolve[0]
        margkx = margsolve[2]
        #corresponding wavenumber
        if rank == 0:
            print('###')
            print('Ra:',margRa)
            print('Ad:',ad_list[i])
            print('kx:',margkx)
            print('###')
        marginalRa.append(margRa)
print(ad_list)
print(marginalRa)
plt.scatter(ad_list,marginalRa)
fulldir = '/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/'
if not os.path.exists(fulldir):
    os.makedirs(fulldir)
plt.tight_layout()
plt.savefig(fulldir+'ad_lower{}ad_upper{}'.format(ad_lower,ad_upper)+'sigs{}'.format(sig_list)+'Nz{}'.format(Nz)+'kx{}'.format(len(kx_global)+1)+'_marginalstabilitycurve.png') 
plt.close()