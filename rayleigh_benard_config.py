
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
if rank == 0:
    print((Nx,Nz))#Nx, Nz = 1024, 256 #4Nx:Nz locked ratio~all powers of 2 Nx, Nz = Nx_prime, Nz_prime
Rayleigh = config.getfloat('param', 'Ra') #CHANGEABLE/Take Notes Lower Number~More turbulence resistant Higher Number~Less turbulence resistant
Prandtl = config.getfloat('param', 'Pr')
kappa = (Rayleigh * Prandtl)**(-1/2) #Thermal dif
paramAD_Diff= config.getboolean('param','isDiff')
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
#Operators
dz = lambda A: d3.Differentiate(A, coords['z'])
dx = lambda A: d3.Differentiate(A, coords['x'])
# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
Q = dist.Field(name='Q', bases=(zbasis, ))
nabad = dist.Field(name='nabad', bases=(zbasis, ))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
integx = lambda arg: d3.Integrate(arg, 'x')
integ = lambda arg: d3.Integrate(integx(arg), 'z')
nu = (Rayleigh / Prandtl)**(-1/2) #viscousity
x, z = dist.local_grids(xbasis, zbasis)
Z = dist.Field(name='Z', bases=(zbasis,))
Z['g'] = z
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

#Decrease functions paramaeters
sig = config.getfloat('param', 'sig')
e = config.getfloat('param', 'e')
Tbump = config.getfloat('param', 'Tbump')
b['g']=2*z-1
b_arr = b.gather_data()
Tplus = b -Tbump + e
Tminus = b -Tbump - e
pi = np.pi
center=config.getfloat("param","center") 
A_dff = 0
A_ad = 0
if paramAD_Diff:
    A_dff = config.getfloat('param','A')
else:
    A_ad = config.getfloat('param','A')

#Adiabat Substitution
adiabat_mean = config.getfloat('param', 'adiabat_mean')   
adiabat_arr = (adiabat_mean+(2/pi)*A_ad*((-pi/2)+(np.arctan(sig*(z-center)**2)))) #Adiabat
nabad['g']=adiabat_arr
#Diffusivity Substitution
    #Horizontal Averaging of diffusivity
if koopa1D == True: 
    Tplus = integx(Tplus/Lx)
    Tminus = integx(Tminus/Lx) 
koopa = kappa*A_dff*(((-pi/2)+np.arctan(sig*Tplus*Tminus))/((pi/2)+np.arctan(sig*e*e)))
temp = (kappa+koopa).evaluate()
temp.change_scales(1) 
temp['g']
koopa_arr = (temp).gather_data() #plotting variables
#Internal Heating
internalheating = ((-np.tanh(50*(z[0,:]-0.9)))-np.tanh(50*((z[0,:])-(1-0.9))))/2 #fucntion
Q['g'] = internalheating
arr_Ad = nabad.gather_data()
arr_Q=Q.gather_data()
arr_z = Z.gather_data()
if rank == 0: 
    maxQ = np.max(arr_Q)
    print(maxQ)
    #diffusivity
    plt.plot(b_arr[0,:],koopa_arr[0,:], color='k', linestyle='solid', linewidth=2, label = "A =" + str(A_dff))
    plt.axhline(y = kappa, color = 'r', linestyle = '--',linewidth=1, label = "A = 0.0")
    plt.xlim(-1,1)
    plt.ylim(0.1,(3/2)*kappa) 
    filename = "/diffusivitybouyprofile.png"
    plt.savefig(name+filename)
    print(name+filename)
    plt.close()
    #adiabat
    plt.plot(arr_z[0,:],arr_Ad[0,:])
    plt.ylim(0,4)
    plt.xlim(0,1)
    filename = "/adaibattempgrad.png"
    plt.savefig(name+filename)
    print(name+filename)
    plt.close()
    #internal heating
    plt.plot(arr_z[0,:],arr_Q[0,:])
    plt.xlim(0,1)
    plt.ylim(-1.05,1.05)
    filename = "/heatingtestfig.png"
    plt.savefig(name+filename)
    print(name+filename)
    plt.close()

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + nabad*(u@ez) + lift(tau_b2) = - u@grad(b) + div(koopa*grad_b) + Q") #Bouyancy equation u@ez supercriticality of 2 
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)") #Momentum equation

#Boundary conditions
problem.add_equation("(b(z=0)) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
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
    # b['g'] += Lz - 2*z # Add linear background
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
            period = 1/10
            amplitude = 1/2
            force_A = 1 +amplitude*np.cos((solver.sim_time*(2*pi))*(period))
            Q['g'][0,:round(Nz/2)]=(force_A)*Q['g'][0,:round(Nz/2)]
            max_Re = flow.max('Re')
            Reynolds_list.append(max_Re)
            time_list.append(solver.sim_time)
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
       
