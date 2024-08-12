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
solvebool=config.getboolean('param','solvbool')
def modesolver (Rayleigh, Prandtl, kx, Nz, ad, sig,Lz,Re):

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
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
    U_c = dist.Field(name='U_c', bases=(zbasis_c,))
    U_r = dist.Field(name='U_r', bases=(zbasis_r,))
    U_z_c = dist.Field(name='U_z_c', bases=(zbasis_c,))
    U_z_r = dist.Field(name='U_z_r', bases=(zbasis_r,))

    z_r  = dist.local_grids(zbasis_r, )[0]
    z_c = dist.local_grids(zbasis_c, )[0]
    ex, ez = coords.unit_vector_fields(dist)
    z_match = Lz/2
    dz = lambda A: d3.Differentiate(A, coords['z'])
    #Substitutions 
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    U_c['g']=(Re*nu*z_c)/(Lz) #Background horz. velocity 
    U_r['g']=(Re*nu*z_r)/(Lz) #Background horz. velocity
    U_z_c['g']=(Re*nu)/(Lz) #Shearing rate
    U_z_r['g']=(Re*nu)/(Lz) #Shearing rate  
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
        #Continuity 
    problem.add_equation("dx(ux_r) + uz_z_r + tau_p = 0")
        #Buoyancy
    problem.add_equation("dt(T_r) + U_r*(dx(T_r)) - kappa*( dx(dx(T_r)) + dz(T_z_r) ) + lift_r(tau_T2_r) - (-nabad_r+2)*uz_r= 0")
        #X comp of momentum
    problem.add_equation("dt(ux_r) + U_r*(dx(ux_r)) + U_z_r*(uz_r) - nu*( dx(dx(ux_r)) + dz(ux_z_r) ) + dx(p_r)     + lift_r(tau_ux2_r)= 0")
        #Z comp of momentum
    problem.add_equation("dt(uz_r) + U_r*(dx(uz_r)) - nu*( dx(dx(uz_r)) + dz(uz_z_r) ) + dz(p_r) - T_r + lift_r(tau_uz2_r) = 0")

    problem.add_equation("T_z_r - dz(T_r) + lift_r(tau_T1_r) = 0")
    problem.add_equation("ux_z_r - dz(ux_r) + lift_r(tau_ux1_r) = 0")
    problem.add_equation("uz_z_r - dz(uz_r) + lift_r(tau_uz1_r) = 0")
    #Bottom Half
    problem.add_equation("dx(ux_c) + uz_z_c = 0")
     
    problem.add_equation("dt(T_c) + U_c*(dx(T_c)) - kappa*( dx(dx(T_c)) + dz(T_z_c) ) + lift_c(tau_T2_c) - (-nabad_c+2)*uz_c= 0")

    problem.add_equation("dt(ux_c) + U_c*(dx(ux_c)) + U_z_c*(uz_c) - nu*( dx(dx(ux_c)) + dz(ux_z_c) ) + dx(p_c)     + lift_c(tau_ux2_c)= 0")
    
    problem.add_equation("dt(uz_c) + U_c*(dx(uz_c)) - nu*( dx(dx(uz_c)) + dz(uz_z_c) ) + dz(p_c) - T_c + lift_c(tau_uz2_c) = 0")

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
    NEV = 40
    target = 3*0.0001
    if solvebool:
        solver.solve_sparse(sp,N=NEV,target=target)    
    else:
        solver.solve_dense(sp)
    return solver
def adiabatresolutionchecker(ad,sig,Nz,Lz,path):
    # Create coordinates and bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64, comm=MPI.COMM_SELF)
    zbasis_r  =  d3.ChebyshevT(coords['z'], size=round(Nz/2), bounds=(Lz/2, Lz), dealias=3/2)
    zbasis_c =  d3.ChebyshevT(coords['z'], size=round(Nz/2), bounds=(0, Lz/2), dealias=3/2)
    z_r  = dist.local_grids(zbasis_r,)[0]
    # print(z_r)
    z_c = dist.local_grids(zbasis_c,)[0]
    # print(z_c)
    scattr_z = np.concatenate((z_c,z_r), axis = 1)
    ad_r = ad-(ad-1)*np.exp(-(z_r-(Lz/2))**2*(1/(2*sig**2)))
    ad_c = ad-(ad-1)*np.exp(-(z_c-(Lz/2))**2*(1/(2*sig**2)))
    scattr_ad = np.concatenate((ad_c,ad_r),axis = 1)
    plt.scatter(scattr_z.T,scattr_ad.T, marker = 'x')
    xbounds=(1/2-1.5*sig,1/2+1.5*sig)
    plt.xlim(xbounds)
    denseZ = np.linspace(xbounds[0],xbounds[1],60000)
    dense_ad = ad-(ad-1)*np.exp(-(denseZ-(Lz/2))**2*(1/(2*sig**2)))
    plt.plot(denseZ,dense_ad, color = 'm')
    plt.axvline(x=1/2-sig, ymin=0, ymax=4, color = 'r', linestyle = '--')
    plt.axvline(x=1/2+sig, ymin=0, ymax=4, color = 'r', linestyle = '--')
    title = 'Adiabat resolution overlay Nz={}'.format(Nz)+r' $\sigma=$'+'{}'.format(sig)
    full_dir = path+'/resolutioncheckplots/'+'sig{}'.format(sig)+'/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    plt.title(title)
    plt.savefig(full_dir+'Nz{}'.format(Nz)+'sig{}'.format(sig)+'adiabatplot.png')
    plt.close()
    return