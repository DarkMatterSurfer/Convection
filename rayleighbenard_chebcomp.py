from docopt import docopt
from configparser import ConfigParser
from dedalus.core import domain
import numpy as np
import h5py
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
import time 
path = os.path.dirname(os.path.abspath(__file__))
# configfile = path +"/options.cfg"
# args = docopt(__doc__)
if len(sys.argv) < 2:
    if not os.path.isfile(path+'/options.cfg'):
        raise
    print('please provide config file')
    configfile = path+'/options.cfg'
else:
    configfile = sys.argv[1]

config = ConfigParser()
config.read(str(configfile))
name = config.get('param','name')
# Parameters
Rayleigh = Rayleigh = config.getfloat('param', 'Ra') 
supercrit=config.getfloat('param','supercrit')
runsupcrit=config.getboolean('param','runsupcrit')
if runsupcrit == True:
    Rayleigh = Rayleigh *supercrit
Prandtl = config.getfloat('param', 'Pr')
Lx, Lz = Lx, Lz = config.getfloat('param', 'Lx'), config.getfloat('param', 'Lz')
z_match = config.getfloat('param','zmatch')
nu = 1/np.sqrt(Rayleigh/Prandtl)
kappa = 1/np.sqrt(Rayleigh*Prandtl)
# n = config.getint('param', 'n')
Nz_prime = config.getint('param','Nz')
Nx_prime = config.getint('param','Nx')
Nx, Nz = Nx_prime, Nz_prime
st=config.getfloat('param', 'st')
maxtimestep=config.getfloat('param', 'maxtimestep')
#Adiabat paramaeterization
sig = config.getfloat('param', 'sig')
ad = config.getfloat('param','back_ad')


# Create coordinates and bases
dtype = np.float64
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
zbasis_r  =  d3.ChebyshevT(coords['z'], size=round(Nz/2), bounds=(z_match, Lz), dealias=3/2)
zbasis_c =  d3.ChebyshevT(coords['z'], size=round(Nz/2), bounds=(0, z_match), dealias=3/2)

# Fields
calib_r=dist.Field(name='calib_r',bases=(xbasis,zbasis_r))
p_r  = dist.Field(name='p_r', bases=(xbasis,zbasis_r))
T_r  = dist.Field(name='T_r', bases=(xbasis,zbasis_r))
u_r  = dist.VectorField(coords, name='u_r', bases=(xbasis,zbasis_r))
nabad_r = dist.Field(name='nabad_r', bases=(zbasis_r, ))
calib_c=dist.Field(name='calib_c',bases=(xbasis,zbasis_c))
p_c  = dist.Field(name='p_c', bases=(xbasis,zbasis_c))
T_c  = dist.Field(name='T_c', bases=(xbasis,zbasis_c))
u_c  = dist.VectorField(coords, name='u_c', bases=(xbasis,zbasis_c))
nabad_c = dist.Field(name='nabad_c', bases=(zbasis_c, ))
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

z_r  = dist.local_grids(zbasis_r,)[0]
z_c = dist.local_grids(zbasis_c,)[0]
x = dist.local_grids(xbasis,)[0]
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
integx = lambda arg: d3.Integrate(arg, 'x')
integ = lambda arg: d3.Integrate(integx(arg), 'z')
#Adiabatic Parameterization
ad_r = ad-(ad-1)*np.exp(-(z_r-(Lz/2))**2*(1/(2*sig**2)))
ad_c = ad-(ad-1)*np.exp(-(z_c-(Lz/2))**2*(1/(2*sig**2)))
nabad_r['g']=ad_r
nabad_c['g']=ad_c
#Variables list
variables = [p_r, T_r, u_r, p_c, T_c, u_c]#, c, p2, T2, u2, c2, p4, T4, u4, c4
taus = [tau_p, tau_T1_r, tau_T2_r, tau_u1_r, tau_u2_r,
        tau_T1_c, tau_T2_c, tau_u1_c, tau_u2_c] #tau_T21, tau_T22, tau_u21, tau_u22, tau_p2,tau_T41, tau_T42, tau_u41, tau_u42, tau_p4

# 2D Boussinesq hydrodynamics
problem = d3.IVP(variables + taus, namespace=locals())

#Top Half
problem.add_equation("trace(grad_u_r) + tau_p = 0")
problem.add_equation("dt(T_r) - kappa*div(grad_T_r) + lift_r(tau_T2_r) +nabad_r*(ez@u_r)= - u_r@grad(T_r)")
problem.add_equation("dt(u_r) - nu*div(grad_u_r) + grad(p_r) + lift_r(tau_u2_r) + ez*(T_r)= - u_r@grad(u_r) ")
#Bottom Half
problem.add_equation("trace(grad_u_c) = 0")
problem.add_equation("dt(T_c) - kappa*div(grad_T_c) + lift_c(tau_T2_c)+nabad_c*(ez@u_c) = - u_c@grad(T_c)")
problem.add_equation("dt(u_c) - nu*div(grad_u_c) + grad(p_c) + lift_c(tau_u2_c) + ez*(T_c) = - u_c@grad(u_c)")
#Matching Conditions
problem.add_equation("p_r(z=z_match) - p_c(z=z_match) = 0")
problem.add_equation("u_r(z=z_match) - u_c(z=z_match) = 0")
problem.add_equation("dz(ex@u_r)(z=z_match) - dz(ex@u_c)(z=z_match) = 0")
problem.add_equation("T_r(z=z_match) - T_c(z=z_match) = 0")
problem.add_equation("dz(T_r)(z=z_match) - dz(T_c)(z=z_match) = 0")
#Boundary Conditions
problem.add_equation("integ(p_r) + integ(p_c) = 0")
problem.add_equation("u_r(z=Lz) = 0")
problem.add_equation("T_r(z=Lz) = -1")
problem.add_equation("u_c(z=0) = 0")
problem.add_equation("T_c(z=0) = 1")

# Build solver
solver = problem.build_solver(d3.RK222)
solver.start_time = time.time()
solver.stop_sim_time = st
logger.info('Solver built')

# Initial conditions
T_r.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise  scales=(0.25, 0.25),
T_c.fill_random('g', seed=5729, distribution='normal', scale=1e-3) # Random noise  scales=(0.25, 0.25),
T_r.change_scales((1, 1))
T_c.change_scales((1, 1))
T_r['g'] *= z_r * (Lz - z_r) # Damp noise at walls
T_r['g'] += Lz - 2*z_r # Add linear background
T_c['g'] *= z_c * (Lz - z_c) # Damp noise at walls
T_c['g'] += Lz - 2*z_c # Add linear background


# Initial timestep
dt = 1e-3

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
Reynolds_num = np.sqrt(u_c@u_c) / nu
flow.add_property(Reynolds_num, name='Re')

#Checkpoints 
# checkpoints = solver.evaluator.add_file_handler(name+'/checkpoints', sim_dt=100, max_writes=1, mode = 'overwrite')
checkpoints = solver.evaluator.add_file_handler(path+'/checkpoints', sim_dt=100, max_writes=1, mode = 'overwrite')
checkpoints.add_tasks(solver.state, layout='g')

#Snapshots Analysis
calib_r['g']=z_r*x
calib_c['g']=z_c*x
# snapshots = solver.evaluator.add_file_handler(name+"/snapshots", sim_dt=0.05, max_writes=50, mode = 'overwrite')
snapshots = solver.evaluator.add_file_handler(path+"/snapshots", sim_dt=0.05, max_writes=50, mode = 'overwrite')
snapshots.add_task(T_r, name='buoyancy_r')
snapshots.add_task(T_c, name='buoyancy_c')
vort_r=-d3.div(d3.skew(u_r))
vort_c=-d3.div(d3.skew(u_c))
snapshots.add_task(vort_r, name='vorticity_r')
snapshots.add_task(vort_c, name='vorticity_c')

#Profiles
    #Reynolds number
# profiles = solver.evaluator.add_file_handler(name+'/profiles', sim_dt=0.0250, max_writes=500, mode = 'overwrite')
profiles = solver.evaluator.add_file_handler(path+'/profiles', sim_dt=0.0250, max_writes=500, mode = 'overwrite')
profiles.add_task(integx(Reynolds_num), name = "reynolds")

#CFL
CFL = d3.CFL(solver, initial_dt=dt, cadence=3, safety=0.35, max_dt=maxtimestep, threshold=0.05)
CFL.add_velocity(u_r)
CFL.add_velocity(u_c)
# Main loop
try:
    logger.info('Starting loop')
    while solver.proceed:

        dt = maxtimestep#CFL.compute_timestep()

        if solver.sim_time > 1:
          dt = min(dt,maxtimestep)

        t_future = solver.sim_time + dt

        solver.step(dt)

        if dt < 1e-10: break
        if (solver.iteration-1) % 10 == 0:
            # logger.info('Iteration: %i, Time: %e, dt: %e,' %(solver.iteration, solver.sim_time, dt))
            # logger.info('Max Re = %f' %flow.max('Re'))
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, dt, max_Re))
            # logger.info('crms = %e' %flow.max('crms'))
            # logger.info('c4rms = %e' %flow.max('c4rms'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()