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
path = os.path.dirname(os.path.abspath(__file__))
from scipy.optimize import minimize_scalar
import time
# configfile = path +"/options.cfg"
# args = docopt(__doc__)
if len(sys.argv) < 2:
    print('please provide config file')
    raise
else:
    configfile = sys.argv[1]
#Config file
config = ConfigParser()
config.read(str(configfile))
name = config.get('param', 'name')
# kappa = (Rayleigh * Prandtl)**(-1/2)
# nu = (Rayleigh / Prandtl)**(-1/2)
#Eigenvalue Spectrum Function
def geteigenval(Rayleigh, Prandtl, kx, Nz, A_ad, adiabat_mean, sig,NEV=10, target=0):
    """Compute maximum linear growth rate."""

    # Parameters
    Nx = 2
    Lx = 2 * np.pi / kx
    Lz = 1
    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
    xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx))
    Xbasis = d3.ComplexFourier(coords['x'], size=64, bounds=(0, Lx))
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz))

    # Fields
    omega = dist.Field(name='omega')
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    mode = dist.Field(name="mode",bases=(Xbasis,))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    nabad = dist.Field(name="nabad",bases=(zbasis, ))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x,X, z = dist.local_grids(xbasis,Xbasis, zbasis)
    Z = dist.Field(name='Z', bases=(zbasis,))
    Z['g'] = z
    arr_z = Z.gather_data()
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction
    mode['g']=np.exp(1j*kx*X) #Eigenmode

    #A~ eigenfunc(A)*e^(ikx-omegat*t)
    dt = lambda A: -1*omega*A #Ansatz for dt

    #Adiabat Parameterization
    pi = np.pi
    adiabat_arr = adiabat_mean-A_ad*(1/sig)/((2*pi)**0.5)*np.exp((-1/2)*(((z-0.5)**2)/sig**2)) #Adiabat
    nabad['g']=adiabat_arr
    arr_Ad = nabad.gather_data()
    # Problem                                                c
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.EVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals(), eigenvalue=omega)
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) - (-nabad+1)*(ez@u) = 0")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = 0")
    problem.add_equation("b(z=0) = 0")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge
    if rank == 0:
        plt.plot(arr_z[0,:],arr_Ad[0,:])
        plt.ylim(0,4)
        plt.xlim(0,1)
        plt.savefig(path+'/'+name+'/adaibattempgrad.png')
        plt.close()
    # Solver
    solver = problem.build_solver(entry_cutoff=0)
    solver.solve_sparse(solver.subproblems[1], NEV, target=target)
    return solver.eigenvalues

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
    counter_tol = 10
    growthrateslist=getgrowthrates(Rayleigh, Prandtl, Nz, A, ad,sig, NEV=10, target = 0)
    if rank == 0:
        print('Intial Growth Rates:',growthrateslist)
    max_omeg = max(growthrateslist)
    results = []
    if rank == 0:
        print('intial parameters maximum growth rate',max_omeg)
    #Finding marginal stability
    A_plus = A+epsilon
    A_minus= A-epsilon
    plusamp_list=getgrowthrates(Rayleigh, Prandtl, Nz, A+epsilon, ad,sig, NEV=10, target = 0)
    ampomeg_plus=max(plusamp_list) 
    minusamp_list=getgrowthrates(Rayleigh, Prandtl, Nz, A-epsilon, ad,sig, NEV=10, target = 0)
    ampomeg_minus=max(minusamp_list)
    omeg_guess = np.inf
    # def rooter(amp_arg):
    #     print('Amplitude:',amp_arg)
    #     omega = max(getgrowthrates(Rayleigh, Prandtl, Nz, amp_arg, ad, NEV=10))
    #     print('Omega:',omega)
    #     return omega
    # result = minimize_scalar(rooter,bounds=(0,2), method='bounded',tol=0.0001)
    # print(result.x)
    # sys.exit()
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
            print('Iteration #:', str(counter) + '\n\n')
            print("ampomeg_plus={}".format(ampomeg_plus))
            print("ampomeg_minus={}".format(ampomeg_minus))
            print("omeg_guess={}".format(omeg_guess))
            print("A={}".format(A_guess))
    if abs(0-omeg_guess) < tol: 
        if rank == 0:
            print(finalrates)
            print(wavenum_list)
        for i in range(len(finalrates)):
            omega_final = finalrates[i]
            if omega_final == omeg_guess:
                maxomeg_kx = wavenum_list[i]
        if rank == 0:
            print("Condtions for marginal stability:")
            print('Rayleigh Number:', Rayleigh)
            print('Prandtl Number:', Prandtl)
            print('Adiabat:', ad)
            print('Amplitude:', A)
            print('Wavenumber (kx) for maximum growth rate:', maxomeg_kx)
        results = [Rayleigh, sig,maxomeg_kx, A]
    return results 

def modesolver(Rayleigh, Prandtl, Nz, adiabat_mean, sig, A_ad):
    print('Mode conditions:\n\n')
    print('Rayleigh:', Rayleigh)
    print('Sig:', sig)
    Lz = 1
    #Wavenumber 
    stability_results = findmarginalomega(Rayleigh,Prandtl,Nz,A_ad,adiabat_mean,sig)
    kx = stability_results[2]
    A_ad = stability_results[3]
    # Bases
    zcoord = d3.Coordinate('z')
    dist = d3.Distributor(zcoord, dtype=np.complex128)
    zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, 1))
    z = dist.local_grid(zbasis)
    arr_x = np.linspace(0,4,256)
    # Fields
    omega = dist.Field(name='omega')
    nabad = dist.Field(name="nabad",bases=(zbasis, ))
    p = dist.Field(name='p', bases=(zbasis,))
    b = dist.Field(name='b', bases=(zbasis,))
    ux = dist.Field(name='ux', bases=(zbasis,))
    uz = dist.Field(name='uz', bases=(zbasis,))
    b_z = dist.Field(name='b_z', bases=(zbasis,))
    ux_z = dist.Field(name='ux_z', bases=(zbasis,))
    uz_z = dist.Field(name='uz_z', bases=(zbasis,))
    arr_x = np.linspace(0,4,256)
    mode=np.exp(1j*kx*arr_x)
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1')
    tau_b2 = dist.Field(name='tau_b2')
    tau_ux1 = dist.Field(name='tau_ux1')
    tau_ux2 = dist.Field(name='tau_ux2')
    tau_uz1 = dist.Field(name='tau_uz1')
    tau_uz2 = dist.Field(name='tau_uz2')

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    dt = lambda A: omega*A
    dx = lambda A: 1j*kx*A
    dz = lambda A: d3.Differentiate(A, zcoord)
    #Adiabat Parameterization
    pi = np.pi
    adiabat_arr = adiabat_mean-A_ad*(1/sig)/((2*pi)**0.5)*np.exp((-1/2)*(((z-0.5)**2)/sig**2))#Adiabat
    nabad['g']=adiabat_arr

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.EVP([p, b, ux, uz, b_z, ux_z, uz_z, tau_p, tau_b1, tau_b2, tau_ux1, tau_uz1, tau_ux2, tau_uz2], namespace=locals(), eigenvalue=omega)
    problem.add_equation("dx(ux) + uz_z + tau_p = 0")
    problem.add_equation("dt(b) - kappa*( dx(dx(b)) + dz(b_z) ) + lift(tau_b2) - (-nabad+1)*uz= 0")
    problem.add_equation("dt(ux) - nu*( dx(dx(ux)) + dz(ux_z) ) + dx(p)     + lift(tau_ux2) = 0")
    problem.add_equation("dt(uz) - nu*( dx(dx(uz)) + dz(uz_z) ) + dz(p) - b + lift(tau_uz2) = 0")
    problem.add_equation("b_z - dz(b) + lift(tau_b1) = 0")
    problem.add_equation("ux_z - dz(ux) + lift(tau_ux1) = 0")
    problem.add_equation("uz_z - dz(uz) + lift(tau_uz1) = 0")
    problem.add_equation("b(z=0) = 0")
    problem.add_equation("ux(z=0) = 0")
    problem.add_equation("uz(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("ux(z=Lz) = 0")
    problem.add_equation("uz(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver()
    sp = solver.subproblems[0]

    solver.solve_dense(sp)
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    evals = evals[np.argsort(-evals.real)]
    print(f"Slowest decaying mode: λ = {evals[0]}")
    solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0])

    b.change_scales(1)
    #Heat Map
    pi=np.pi
    phase=1
    phaser=np.exp(((1j*phase)*(2*pi))/4)
    #Modes
    b_mode=(np.outer(b['g'],mode)*phaser).real
    return b_mode

# Parameters
Nz = 64
Rayleigh = config.getfloat('param', 'Ra') 
Prandtl = config.getfloat('param', 'Pr')
<<<<<<< HEAD
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
kx_global = np.linspace(0.001, 4, 10)
=======
kx_global = np.linspace(0.001, 4, 5)
>>>>>>> de99828be2b573529be9167aa7b58bcd2eee7154
wavenum_list = []
for i in kx_global:
    wavenum_list.append(i)
maxomeg_kx = 0
if rank == 0:
    print('Wavenumbers :',wavenum_list)
NEV = 1
A = config.getfloat('param','A')
sig = config.getfloat('param','sig')
ad = config.getfloat('param', 'adiabat_mean')  
epsilon = 0.1
#Search parameters
tol = 0.00001

#Plotting
# Bases
zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord, dtype=np.complex128)
zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, 1))
z = dist.local_grid(zbasis)
arr_x = np.linspace(0,4,256)

#Top left corner
fig, axs = plt.subplots(2, 2)
ax = axs[0, 0]
c = ax.pcolor(arr_x,z,modesolver(Rayleigh,Prandtl,Nz,ad,sig,A), cmap='RdBu') 
ax.set_ylabel('Ra='+str(Rayleigh))
fig.tight_layout()
fig.colorbar(c, ax=ax)
#Top right corner
Rayleigh=1e3
sig=0.02
ax = axs[0, 1]
c = ax.pcolor(arr_x,z,modesolver(Rayleigh,Prandtl,Nz,ad,sig,A),cmap='RdBu') 
fig.tight_layout()
fig.colorbar(c, ax=ax)
#Bottom left corner
Rayleigh=1710
sig=0.01
ax = axs[1, 0]
c = ax.pcolor(arr_x,z,modesolver(Rayleigh,Prandtl,Nz,ad,sig,A), cmap='RdBu')
ax.set_ylabel('Ra='+str(Rayleigh))
ax.set_xlabel(r'$\sig=\$'+str(sig))
fig.tight_layout()
fig.colorbar(c, ax=ax)
#Bottom right corner
Rayleigh=1710
sig=0.02
ax = axs[1, 1]
c = ax.pcolor(arr_x,z,modesolver(Rayleigh,Prandtl,Nz,ad,sig,A), cmap='RdBu')
ax.set_xlabel(r'$\sig=\$'+str(sig))
fig.colorbar(c, ax=ax)

folderstring= "Ra"+str(Rayleigh)+"Pr"+str(Prandtl)
fig.tight_layout()
plt.savefig(path+"/multipanelheatmode.png")
if rank == 0:
    print(path+"/multipanelheatmode.png")
plt.close()