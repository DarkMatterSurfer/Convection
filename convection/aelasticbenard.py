"""
Rayleigh-Benard 2d simulation

Usage:
    rayleigh_benard.py [--ra=<float> --pr=<float> --st=<float> --lx=<float> --sn=<string> --state=<string>]

Options:
    --ra=<float>  rayleigh number [default: 2e6]
    --pr=<float>  prandtl number [default: 1]
    --st=<float>  stop_sim [default: 50]
    --lx=<float>  lx [default: 4]
    --sn=<string>  output directory for snapshots [default: snapshots]
    --state=<string>  load directory for previous state [default: none]

"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import mpi4py
import matplotlib.pyplot as plt
from docopt import docopt
import sys

args = docopt(__doc__)

CW = mpi4py.MPI.COMM_WORLD
ncores = CW.size
#Multi-Rayleigh Runs
# Ra_list = [1e4,2e4,4e4,1e5,2e5,4e5,1e6,2e6,]
# for index in Ra_list:
# Parameters
Lx, Lz = float(args['--lx']), 1
n = 6 #power of two 
Nz_prime = 2**n
Nx_prime = float(args['--lx']) * Nz_prime
Nx, Nz = Nx_prime, Nz_prime #Nx, Nz = 1024, 256 #4Nx:Nz locked ratio~all powers of 2 Nx, Nz = Nx_prime, Nz_prime
Rayleigh = float(args['--ra']) #CHANGEABLE/Take Notes Lower Number~More turbulence resistant Higher Number~Less turbulence resistant
Prandtl = float(args['--pr'])
dealias = 3/2
stop_sim_time = float(args['--st'])
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
s = dist.Field(name='s', bases=(xbasis,zbasis)) # entropy
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_s1 = dist.Field(name='tau_s1', bases=xbasis)
tau_s2 = dist.Field(name='tau_s2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
# Hollow Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2) #Thermal dif

sig = 30 
e = 0.3
Tbump = 0.5
Tplus = s -Tbump + e
Tminus = s -Tbump - e
A = 0.0
pi = np.pi

koopa = kappa*A*(((-pi/2)+np.arctan(sig*Tplus*Tminus))/((pi/2)+np.arctan(sig*e*e)))


nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_s = d3.grad(s) + ez*lift(tau_s1) # First-order reduction
#background quantities
rho = dist.Field(name='rho', bases=(zbasis, ))
Q = dist.Field(name='Q', bases=(zbasis, ))
phi = dist.Field(name='phi', bases=(zbasis, ))
T = dist.Field(name='T', bases=(zbasis, ))

gamma = 5/3 #Neutral Hydrogen
cp = gamma/(gamma-1)
cv = 1/(gamma-1)
g = 1 #paramter
q = ((gamma-1)/gamma)*g
T_b = 1 
T['g'] = q*z + T_b
rho['g'] = np.power(q*z + T_b, (1/(1-gamma)))
grad_rho = d3.grad(rho)
z_bl = 0.1
Q['g'] = 1 - np.greater(z, z_bl) - np.greater(z, 1 - z_bl)

cond_flux = grad_s @ ez
bottom_flux = d3.Integrate(cond_flux(z=0), 'x')
top_flux = d3.Integrate(cond_flux(z=Lz), 'x')

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, s, u, tau_p, tau_s1, tau_s2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("rho*trace(grad_u) + tau_p + u@grad_rho = 0") #
problem.add_equation("dt(s) + lift(tau_s2) = - u@grad(s) - div(Frad)/(rho*T) + (PHI+Q)/(rho*T)") #Bouyancy equation u@ez supercriticality of 2 
problem.add_equation("dt(u) + grad(p) - s*ez*g/cp + lift(tau_u2) = - u@grad(u) + div(PI)/rho") #Momentum equation
problem.add_equation("s(z=0) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("s(z=Lz) = 0")
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
state = args['--state']
if (state == 'none' ):
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise intal parameters~
    b['g'] *= z * (Lz - z) # Damp noise at walls
    b['g'] += Lz - z # Add linear background
else:
    solver.load_state(state)
    solver.sim_time = 0.0

# Analysis
snapshots = solver.evaluator.add_file_handler(args['--sn'], sim_dt=0.05, max_writes=50)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

checkpoint = solver.evaluator.add_file_handler('checkpoint_' + args['--sn'], sim_dt=199)
checkpoint.add_tasks(solver.state, layout='g')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.05,
            max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Plotting Lists
Reynolds_list = [] #max Reynolds number list
time_list = []  #time list

heatflux_top = []
heatflux_bottom = []
#Main Loop 
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            f1 = np.average(np.mean(bottom_flux.evaluate()['g']))
            f2 = np.average(np.mean(top_flux.evaluate()['g']))
            max_Re = flow.max('Re')
            Reynolds_list.append(max_Re)
            time_list.append(solver.sim_time)
            heatflux_top.append(f2)
            heatflux_bottom.append(f1)
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, flux_bottom=%f, flux_top=%f' %(solver.iteration, solver.sim_time, timestep, max_Re, f1, f2))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

if (CW.rank == 0):
    filetype = '.csv'
    fluxtop = 'topflux'
    fluxbottom = 'bottomflux'
    reyrey = 'Reynolds'
    Run_num = ''

    #Declaring file name 
    Reynolds_File = open(str(Rayleigh)+"Run" + Run_num + reyrey + filetype, 'w') 
    TopFlux_File = open(str(Rayleigh)+'Run' + Run_num + fluxtop + filetype, 'w')
    BottomFlux_File = open(str(Rayleigh)+'Run' + Run_num + fluxbottom + filetype, 'w')


    #Writing in file
    for i in range(len(Reynolds_list)):
        Reynolds_File.write(str(time_list[i]) + ', ' + str(Reynolds_list[i] ) + '\n')
        
    for i in range(len(heatflux_top)):
        TopFlux_File.write(str(time_list[i]) + ', ' + str(heatflux_top[i] ) + '\n')

    for i in range(len(heatflux_bottom)):
        BottomFlux_File.write(str(time_list[i]) + ', ' + str(heatflux_bottom[i] ) + '\n')

    # Reynolds_File.close()
    TopFlux_File.close()
    BottomFlux_File.close()
    # plt.plot(time_list, Reynolds_list)
    # plt.title('Simulated Maximum Reynolds Number in Time-domain' + '\n' + 'Ra=' + str(Rayleigh), fontsize = 10)
    # plt.xlabel('TIme')
    # plt.ylabel('Reynolds [Re]')
    # plt.show()
