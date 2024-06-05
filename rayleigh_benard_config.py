
from docopt import docopt
from configparser import ConfigParser
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import mpi4py
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

config = ConfigParser()
config.read(str(configfile))
CW = mpi4py.MPI.COMM_WORLD
ncores = CW.size
# Parameters
Lx, Lz = config.getfloat('param', 'Lx'), config.getfloat('param', 'Lz')
n = config.getint('param', 'n') #power of two 
Nz_prime = 2**n
Nx_prime = config.getint('param','aspect') * Nz_prime #float(args['--lx']) 
Nx, Nz = Nx_prime, Nz_prime
print((Nx,Nz))#Nx, Nz = 1024, 256 #4Nx:Nz locked ratio~all powers of 2 Nx, Nz = Nx_prime, Nz_prime
Rayleigh = config.getfloat('param', 'Ra') #CHANGEABLE/Take Notes Lower Number~More turbulence resistant Higher Number~Less turbulence resistant
Prandtl = config.getfloat('param', 'Pr')
adiabat = config.getfloat('param', 'adiabat')
dealias = 3/2
stop_sim_time = config.getfloat('param', 'st')
timestepper = d3.RK222
max_timestep = config.getfloat('param', 'maxtimestep')
dtype = np.float64
name = config.get('param', 'name')
koopa1D= config.getboolean('param','koopa1D')
# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
integx = lambda arg: d3.Integrate(arg, 'x')
integ = lambda arg: d3.Integrate(integx(arg), 'z')
# Hollow Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2) #Thermal dif

sig = config.getfloat('param', 'sig')
e = config.getfloat('param', 'e')
Tbump = config.getfloat('param', 'Tbump')
Tplus = b -Tbump + e
Tminus = b -Tbump - e
if koopa1D == True: 
    Tplus = integx(Tplus/Lx)
    Tminus = integx(Tminus/Lx)

A = config.getfloat('param', 'A')
pi = np.pi
koopa = kappa*A*(((-pi/2)+np.arctan(sig*Tplus*Tminus))/((pi/2)+np.arctan(sig*e*e)))


nu = (Rayleigh / Prandtl)**(-1/2) #viscousity
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction


# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + adiabat*(u@ez) + lift(tau_b2) = - u@grad(b) + div(koopa*grad_b)") #Bouyancy equation u@ez supercriticality of 2 
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)") #Momentum equation
#Boundary conditions
problem.add_equation("b(z=0) = Lz")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = -Lz")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# grad_ => gradient
# div => divergence
# lap => laplacian
# @ => dot product


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
state = config.get('param', 'state')
if (state == 'none' ):
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise intal parameters~
    b['g'] *= z * (Lz - z) # Damp noise at walls
    b['g'] += Lz - 2*z # Add linear background
else:
    solver.load_state(state)
    solver.sim_time = 0.0

#Checkpoints 
checkpoints = solver.evaluator.add_file_handler(name+'/checkpoints', sim_dt=100, max_writes=1, mode = 'overwrite')
checkpoints.add_tasks(solver.state, layout='g')

# Analysis
snapshots = solver.evaluator.add_file_handler(name+"/snapshots", sim_dt=0.05, max_writes=50, mode = 'overwrite')
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

#Profiles 
profiles = solver.evaluator.add_file_handler(name+'/profiles', sim_dt=0.0250, max_writes=500, mode = 'overwrite')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
Reynolds_Num = (np.sqrt(u@u))/(nu)
flow.add_property(Reynolds_Num, name='Re') #Reynolds Number
profiles.add_task(integx(Reynolds_Num), name = "reynolds")
profiles.add_task(integ(Reynolds_Num), name = "reynolds_timeaveraged")
#Fluxes
    #Convective Flux
profiles.add_task(integx(u@ez * b), name = 'convectiveflx') #Convective flux is <w*b>
    #Diffusive Flux
profiles.add_task(integx(-(kappa + koopa)* grad_b@ez), name = 'diffusiveflx') #diffusive flux is <-kappa*dz(b)>
Ke_x = ((u@ex)**2)/2
profiles.add_task(integx(abs(-(kappa + koopa)* grad_b@ex)), name = 'Xdiff')

#Nusselt Number
Nusselt =(integ(u@ez*b)+(integ(-kappa * grad_b@ez)))/integ(-kappa * grad_b@ez)
profiles.add_task(Nusselt, name = "Nusselt") 

# Buoyancy 
profiles.add_task(integx(b), name = "buoyancy")
#Kinetic Energy
Ke_z = ((u@ez)**2)/2
profiles.add_task(integx(Ke_x),name = 'kex')
profiles.add_task(integx(Ke_z),name = 'kez')
profiles.add_task(integ(Ke_x),name = 'kex_whole')
profiles.add_task(integ(Ke_z),name = 'kez_whole')
#Mean Temperature Profile 
profiles.add_task(integx(b),name = 'mean_temp')

# Velocities
profiles.add_task(integx(u@ex), name = 'mean_u@x')
profiles.add_task(integx(u@ez), name = 'mean_u@z')

#Enstrophy
vort = d3.div(d3.skew(u))
profiles.add_task(integx(vort**2), name = 'entrsophy')

# CFLp
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.05,
            max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)



# Plotting Lists
Reynolds_list = [] #max Reynolds number list
time_list = []  #time list

# heatflux_top = []
# heatflux_bottom = []
#Main Loop 
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            Reynolds_list.append(max_Re)
            time_list.append(solver.sim_time)
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

