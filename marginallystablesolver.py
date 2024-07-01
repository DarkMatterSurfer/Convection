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
from docopt import docopt
import sys
import os 
path = os.path.dirname(os.path.abspath(__file__))
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

#Eigenvalue Spectrum Function
def geteigenval(Rayleigh, Prandtl, kx, Nz, A, ad, NEV=10, target=0):
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
    adiabat_mean = ad
    pi = np.pi
    A_ad = A
    sig = 0.01
    adiabat_arr = adiabat_mean-A_ad*(1/sig)/((2*pi)**0.5)*np.exp((-1/2)*(((z-0.5)**2)/sig**2)) #Adiabat
    nabad['g']=adiabat_arr
    arr_Ad = nabad.gather_data()
    # Problem
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

    plt.plot(arr_z[0,:],arr_Ad[0,:])
    plt.ylim(0,4)
    plt.xlim(0,1)
    plt.show()
    plt.savefig('/home/iiw7750/Convection/adaibattempgrad.png')
    plt.close()
    # Solver
    solver = problem.build_solver(entry_cutoff=0)
    solver.solve_sparse(solver.subproblems[1], NEV, target=target)
    return solver.eigenvalues

def getgrowthrates(Rayleigh, Prandtl,Nz, A, ad, NEV=10):
    if __name__ == "__main__":

            import time
            import matplotlib.pyplot as plt
            comm = MPI.COMM_WORLD
            # Compute growth rate over local wavenumbers
            kx_local = kx_global[comm.rank::comm.size]
            t1 = time.time()
            # for all 
            growth_locallist = []
            frequecny_locallist = []
            for kx in kx_local:
                eigenvals = geteigenval(Rayleigh, Prandtl, kx, Nz,A,ad, NEV=NEV) #np.array of complex
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
            for i in growth_global:
                ratelist.append(i)
    return ratelist
# Parameters
Nz = 64
Rayleigh = config.getfloat('param', 'Ra') 
Prandtl = config.getfloat('param', 'Pr')
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
kx_global = np.linspace(0.001, 4, 50)
wavenum_list = []
for i in kx_global:
    wavenum_list.append(i)
maxomeg_kx = 0
# if rank == 0:
#     print(wavenum_list)
NEV = 1
A = config.getfloat('param','A')
ad = config.getfloat('param', 'adiabat_mean')  
epsilon = 0.05
#Search parameters
tol = 0.01
counter = 0
counter_tol = 10
growthrateslist=getgrowthrates(Rayleigh, Prandtl, Nz, A, ad, NEV=10)
# if rank == 0:
#     print(growthrateslist)
max_omeg = max(growthrateslist)
if rank == 0:
    print('intial parameters maximum growth rate',max_omeg)
#Finding marginal stability
while abs(0-max_omeg) > tol:
    plusamp_list=getgrowthrates(Rayleigh, Prandtl, Nz, A+epsilon, ad, NEV=10)
    ampomeg_plus=max(plusamp_list)
    minusamp_list=getgrowthrates(Rayleigh, Prandtl, Nz, A-epsilon, ad, NEV=10)
    ampomeg_minus=max(minusamp_list)
    if abs(0-max_omeg)>abs(0-ampomeg_plus):
        A = (A+epsilon+A)/2
        finalrates = plusamp_list
        max_omeg = ampomeg_plus
        if rank == 0:
            print('New max omega:', str(max_omeg)+'\n\n')
    elif abs(0-max_omeg)>abs(0-ampomeg_minus):
        A = (A-epsilon+A)/2
        finalrates = minusamp_list
        max_omeg = ampomeg_minus
        if rank == 0:
            print('New max omega:', str(max_omeg)+'\n\n')
            
    if rank == 0:
        print('Iteration #:', str(counter) + '\n\n')
    counter=counter+1
    if abs(0-max_omeg)<tol:
        for i in range(len(finalrates)):
            if max(finalrates) == max_omeg:
                maxomeg_kx = wavenum_list[i]
        if rank == 0:
            print("Condtions for marginal stability:")
            print('Rayleigh Number:', Rayleigh)
            print('Prandtl Number:', Prandtl)
            print('Adiabat:', ad)
            print('Amplitude:', A)
            print('Wavenumber (kx) for maximum growth rate:', maxomeg_kx)