"""
Dedalus script for calculating the maximum linear growth rates in no-slip
Rayleigh-Benard convection over a range of horizontal wavenumbers. This script
demonstrates solving a 1D eigenvalue problem in a Cartesian domain. It can
be ran serially or in parallel, and produces a plot of the highest growth rate
found for each horizontal wavenumber. It should take a few seconds to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard_evp.py
"""

import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
import sys
import mpi4py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = logging.getLogger(__name__)

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


if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt
    comm = MPI.COMM_WORLD

    # Parameters
    Nz = 64
    Rayleigh = 1000
    Prandtl =  1
    kx_global = np.linspace(0.001, 4, 50)
    NEV = 1

    A = 0.4
    ad = 0
    # Compu te growth rate over local wavenumbers
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

    # Plot growth rates from root process
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
        plt.savefig("/home/iiw7750/Convection/eigenvalprob_plots/"+"Ra"+str(Rayleigh)+"Pr"+str(Prandtl)+"/"+filename)
        print(filename)
        plt.close()