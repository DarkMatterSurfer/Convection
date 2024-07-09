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
#Eigenvalue Spectrum Function
def geteigenval(Rayleigh, Prandtl, kx, Nz, A_ad, adiabat_mean, sig,Lz=1,NEV=10, target=0):
    """Compute maximum linear growth rate."""

    # Parameters
    Nx = 2
    Lx = 2 * np.pi / kx
    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
    xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx))

    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz))

    # Fields
    omega = dist.Field(name='omega')
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    # mode = dist.Field(name="mode",bases=(Xbasis,))
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
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction
    # mode['g']=np.exp(1j*kx*X) #Eigenmode

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

    # Solver
    solver = problem.build_solver(entry_cutoff=0)
    solver.solve_sparse(solver.subproblems[1], NEV, target=target)
    return solver.eigenvalues

def modesolver (Rayleigh, Prandtl, kx, Nz, A_ad, adiabat_mean, sig,Lz,NEV=10, target=0):

    # Bases
    zcoord = d3.Coordinate('z')
    dist = d3.Distributor(zcoord, dtype=np.complex128)
    zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, Lz))
    z = dist.local_grid(zbasis)
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
    return solver