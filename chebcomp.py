import numpy as np
from mpi4py import MPI
import time
import h5py

from dedalus import public as d3

import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = (1, 1)
z_match = 0.5

Rayleigh = 1e3
Prandtl = 1.
nu = 1/np.sqrt(Rayleigh/Prandtl)
kappa = 1/np.sqrt(Rayleigh*Prandtl)

S = 100
T_top = -60

N2 = np.abs(S*T_top*2)
f = np.sqrt(N2)/(2*np.pi)

N = 64
dtype = np.float64

# Create coordinates and bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis      = d3.RealFourier(coords['x'], size=N, bounds=(-Lx/2, Lx/2), dealias=3/2)
zbasis_r  =  d3.ChebyshevT(coords['z'], size=int(N/2), bounds=(z_match, Lz), dealias=3/2)
zbasis_c =  d3.ChebyshevT(coords['z'], size=N, bounds=(0, z_match), dealias=3/2)

# Fields
p_r  = dist.Field(name='p_r', bases=(xbasis,zbasis_r))
T_r  = dist.Field(name='T_r', bases=(xbasis,zbasis_r))
u_r  = dist.VectorField(coords, name='u_r', bases=(xbasis,zbasis_r))
c    = dist.Field(name='c', bases=(xbasis, zbasis_r))
c2   = dist.Field(name='c2', bases=(xbasis, zbasis_r))
p2   = dist.Field(name='p2', bases=(xbasis,zbasis_r))
T2   = dist.Field(name='T2', bases=(xbasis,zbasis_r))
u2   = dist.VectorField(coords, name='u2', bases=(xbasis,zbasis_r))
c4   = dist.Field(name='c4', bases=(xbasis, zbasis_r))
p4   = dist.Field(name='p4', bases=(xbasis,zbasis_r))
T4   = dist.Field(name='T4', bases=(xbasis,zbasis_r))
u4   = dist.VectorField(coords, name='u4', bases=(xbasis,zbasis_r))
p_c  = dist.Field(name='p_c', bases=(xbasis,zbasis_c))
T_c  = dist.Field(name='T_c', bases=(xbasis,zbasis_c))
u_c  = dist.VectorField(coords, name='u_c', bases=(xbasis,zbasis_c))
tau_p  = dist.Field(name='tau_p')
tau_p2 = dist.Field(name='tau_p2')
tau_p4 = dist.Field(name='tau_p4')
tau_T1_r  = dist.Field(name='tau_T1_r', bases=xbasis)
tau_T2_r  = dist.Field(name='tau_T2_r', bases=xbasis)
tau_u1_r  = dist.VectorField(coords, name='tau_u1_r', bases=xbasis)
tau_u2_r  = dist.VectorField(coords, name='tau_u2_r', bases=xbasis)
tau_T1_c = dist.Field(name='tau_T1_c', bases=xbasis)
tau_T2_c = dist.Field(name='tau_T2_c', bases=xbasis)
tau_u1_c = dist.VectorField(coords, name='tau_u1_c', bases=xbasis)
tau_u2_c = dist.VectorField(coords, name='tau_u2_c', bases=xbasis)
tau_T21  = dist.Field(name='tau_T21', bases=xbasis)
tau_T22  = dist.Field(name='tau_T22', bases=xbasis)
tau_u21  = dist.VectorField(coords, name='tau_u21', bases=xbasis)
tau_u22  = dist.VectorField(coords, name='tau_u22', bases=xbasis)
tau_T41  = dist.Field(name='tau_T41', bases=xbasis)
tau_T42  = dist.Field(name='tau_T42', bases=xbasis)
tau_u41  = dist.VectorField(coords, name='tau_u41', bases=xbasis)
tau_u42  = dist.VectorField(coords, name='tau_u42', bases=xbasis)

x, z_r  = dist.local_grids(xbasis, zbasis_r)
x, z_c = dist.local_grids(xbasis, zbasis_c)
ex, ez = coords.unit_vector_fields(dist)
dz = lambda A: d3.Differentiate(A, coords['z'])
lift_basis_r  = zbasis_r.derivative_basis(1)
lift_basis_c = zbasis_c.derivative_basis(1)
lift_r  = lambda A: d3.Lift(A, lift_basis_r, -1)
lift_c = lambda A: d3.Lift(A, lift_basis_c, -1)
grad_u_r = d3.grad(u_r) + ez*lift_r(tau_u1_r) # First-order reduction
grad_T_r = d3.grad(T_r) + ez*lift_r(tau_T1_r) # First-order reduction
grad_u_c = d3.grad(u_c) + ez*lift_c(tau_u1_c) # First-order reduction
grad_T_c = d3.grad(T_c) + ez*lift_c(tau_T1_c) # First-order reduction

grad_u2 = d3.grad(u2) + ez*lift_r(tau_u21) # First-order reduction
grad_T2 = d3.grad(T2) + ez*lift_r(tau_T21) # First-order reduction
grad_u4 = d3.grad(u4) + ez*lift_r(tau_u41) # First-order reduction
grad_T4 = d3.grad(T4) + ez*lift_r(tau_T41) # First-order reduction

alpha_r  = dist.Field(bases=(xbasis, zbasis_r))
alpha_c = dist.Field(bases=(xbasis, zbasis_c))
alpha_r.preset_scales(3/2)
alpha_c.preset_scales(3/2)
alpha_r.require_grid_space()
alpha_c.require_grid_space()

def rho_r_func(*args): # args[0] is T
    np.place(alpha_r.data, args[0].data>0, -1)
    np.place(alpha_r.data, args[0].data<=0, S)
    return alpha_r.data*args[0].data

def rho_c_func(*args): # args[0] is T
    np.place(alpha_c.data, args[0].data>0, -1)
    np.place(alpha_c.data, args[0].data<=0, S)
    return alpha_c.data*args[0].data

def rho_r(*args):
    return d3.GeneralFunction(dist, T_r.domain, layout='g', tensorsig=T_r.tensorsig, dtype=dtype, func=rho_r_func, args=args)

def rho_c(*args):
    return d3.GeneralFunction(dist, T_c.domain, layout='g', tensorsig=T_c.tensorsig, dtype=dtype, func=rho_c_func, args=args)

f_N = 30
z_d = 0.925
Dz_d = 0.025

damping = dist.Field(bases=zbasis_r)
damping['g'] = 0.5*f_N*( 1 + np.tanh( (z_r-z_d)/Dz_d ) )
damping = d3.Grid(damping)

NL = dist.Field(bases=zbasis_r)
NL['g'] = 0.5*(1 + np.tanh( (z_r-0.65)/0.01 ))
NL = d3.Grid(NL)

c0 = dist.Field(bases=zbasis_r)
c0['g'] = z_r**2
c0 = d3.Grid(c0)

variables = [p_r, T_r, u_r, p_c, T_c, u_c, c, p2, T2, u2, c2, p4, T4, u4, c4]
taus = [tau_p, tau_T1_r, tau_T2_r, tau_u1_r, tau_u2_r,
        tau_T1_c, tau_T2_c, tau_u1_c, tau_u2_c,
        tau_T21, tau_T22, tau_u21, tau_u22, tau_p2,
        tau_T41, tau_T42, tau_u41, tau_u42, tau_p4]

# 2D Boussinesq hydrodynamics
problem = d3.IVP(variables + taus, namespace=locals())
problem.add_equation("dt(c)  = -u_r@grad(c)*NL")
problem.add_equation("dt(c2) =  -u2@grad(c2)*NL")
problem.add_equation("dt(c4) =  -u4@grad(c4)*NL")

problem.add_equation("trace(grad_u2) + tau_p2 = 0")
problem.add_equation("dt(T2) - kappa*div(grad_T2) + lift_r(tau_T22) = - u2@grad(T2) - u2@grad(integ(T_r,'x')/Lx)")
#problem.add_equation("dt(T2_rad) - kappa*div(grad_T2_rad) + lift_rad(tau_T22_rad) = - u2_rad@grad(T2_rad)")
problem.add_equation("dt(u2) - nu*div(grad_u2) + grad(p2) + lift_r(tau_u22) = - u2@grad(u2) - ez*S*T2 - u2*damping")

problem.add_equation("trace(grad_u4) + tau_p4 = 0")
problem.add_equation("dt(T4) - kappa*div(grad_T4) + lift_r(tau_T42) = - u4@grad(T4) - u4@grad(integ(T_r,'x')/Lx)")
problem.add_equation("dt(u4) - nu*div(grad_u4) + grad(p4) + lift_r(tau_u42) = - u4@grad(u4) - ez*S*T4 - u4*damping")

problem.add_equation("trace(grad_u_r) + tau_p = 0")
problem.add_equation("dt(T_r) - kappa*div(grad_T_r) + lift_r(tau_T2_r) = - u_r@grad(T_r)")
problem.add_equation("dt(u_r) - nu*div(grad_u_r) + grad(p_r) + lift_r(tau_u2_r) = - u_r@grad(u_r) - ez*rho_r(T_r) - u_r*damping")
problem.add_equation("trace(grad_u_c) = 0")
problem.add_equation("dt(T_c) - kappa*div(grad_T_c) + lift_c(tau_T2_c) = - u_c@grad(T_c)")
problem.add_equation("dt(u_c) - nu*div(grad_u_c) + grad(p_c) + lift_c(tau_u2_c) = - u_c@grad(u_c) - ez*rho_c(T_c)")

problem.add_equation("p_r(z=z_match) - p_c(z=z_match) = 0")
problem.add_equation("u_r(z=z_match) - u_c(z=z_match) = 0")
problem.add_equation("dz(ex@u_r)(z=z_match) - dz(ex@u_c)(z=z_match) = 0")
problem.add_equation("T_r(z=z_match) - T_c(z=z_match) = 0")
problem.add_equation("dz(T_r)(z=z_match) - dz(T_c)(z=z_match) = 0")

problem.add_equation("u2(z=z_match) - 2*u_r(z=z_match) = 0")
problem.add_equation("T2(z=z_match) - 2*(T_r - integ(T_r,'x')/Lx)(z=z_match) = 0")
#problem.add_equation("T2_rad(z=z_match) - T_rad(z=z_match) = 0")
problem.add_equation("ez@u2(z=Lz) = 0")
problem.add_equation("dz(ex@u2)(z=Lz) = 0")
problem.add_equation("T2(z=Lz) = 0")
#problem.add_equation("T2_rad(z=Lz) = T_top")
problem.add_equation("integ(p2) = 0")

problem.add_equation("u4(z=z_match) - 4*u_r(z=z_match) = 0")
problem.add_equation("T4(z=z_match) - 4*(T_r - integ(T_r,'x')/Lx)(z=z_match) = 0")
problem.add_equation("ez@u4(z=Lz) = 0")
problem.add_equation("dz(ex@u4)(z=Lz) = 0")
problem.add_equation("T4(z=Lz) = 0")
problem.add_equation("integ(p4) = 0")

problem.add_equation("integ(p_r) + integ(p_c) = 0")
problem.add_equation("ez@u_r(z=Lz) = 0")
problem.add_equation("dz(ex@u_r)(z=Lz) = 0")
problem.add_equation("T_r(z=Lz) = T_top")
problem.add_equation("ez@u_c(z=0) = 0")
problem.add_equation("dz(ex@u_c)(z=0) = 0")
problem.add_equation("T_c(z=0) = 1")

# Build solver
solver = problem.build_solver(d3.RK222)
solver.start_time = time.time()
solver.stop_sim_time = 100001.
logger.info('Solver built')

# Initial conditions

c['g'] = z_r**2
c2['g'] = z_r**2
c4['g'] = z_r**2

T_r.fill_random('g', seed=42, distribution='normal', scales=(0.25, 0.25), scale=5e-3) # Random noise
T_c.fill_random('g', seed=5729, distribution='normal', scales=(0.25, 0.25), scale=5e-3) # Random noise
T_r.change_scales((1, 1))
T_c.change_scales((1, 1))

z_i = 0.5
del_z = 0.02

T_r['g'] *= (z_r*(Lz-z_r)) * 0.5*(1 - np.tanh( (z_r - 0.4)/del_z ))
T_c['g'] *= (z_c*(Lz-z_c)) * 0.5*(1 - np.tanh( (z_c - 0.4)/del_z ))

T_func = lambda z: del_z*np.log(np.cosh( (z-z_i)/del_z ))

T_r['g'] += -1*(z_r - T_func(z_r) - 1 + T_func(0) ) + T_top*(z_r + T_func(z_r) - T_func(0))
T_c['g'] += -1*(z_c - T_func(z_c) - 1 + T_func(0) ) + T_top*(z_c + T_func(z_c) - T_func(0))

#T2_rad.change_scales(3/2)
#T_rad.change_scales(3/2)
#T2_rad['g'] = T_rad['g']

# Initial timestep
dt = 1e-3
max_dt = 0.0005

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_c@u_c) / nu, name='Re')
flow.add_property(np.sqrt((c-c0)*(c-c0)), name='crms')
flow.add_property(np.sqrt((c4-c0)*(c4-c0)), name='c4rms')

CFL = d3.CFL(solver, initial_dt=dt, cadence=3, safety=0.35, max_dt=max_dt, threshold=0.05)
CFL.add_velocity(u_c)

# Main loop
try:
    logger.info('Starting loop')
    while solver.proceed:

        dt = CFL.compute_timestep()

        if solver.sim_time > 1:
          dt = min(dt,max_dt)

        t_future = solver.sim_time + dt

        solver.step(dt)

        if dt < 1e-10: break
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
            logger.info('crms = %e' %flow.max('crms'))
            logger.info('c4rms = %e' %flow.max('c4rms'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()