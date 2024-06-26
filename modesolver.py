import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import time
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD

# Parameters
Nz = 64
Rayleigh = 1000000
Prandtl = 1
kx = 3.75
NEV = 10
Lz = 1
target = 0

# Bases
zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord, dtype=np.complex128)
zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, Lz))
z = dist.local_grid(zbasis)
# Fields
omega = dist.Field(name='omega')
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

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.EVP([p, b, ux, uz, b_z, ux_z, uz_z, tau_p, tau_b1, tau_b2, tau_ux1, tau_uz1, tau_ux2, tau_uz2], namespace=locals(), eigenvalue=omega)
problem.add_equation("dx(ux) + uz_z + tau_p = 0")
problem.add_equation("dt(b) - kappa*( dx(dx(b)) + dz(b_z) ) + lift(tau_b2) - uz = 0")
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
p.change_scales(1)
ux.change_scales(1)
uz.change_scales(1)
#Heat Map
pi=np.pi
phase=1
phaser=np.exp(((1j*phase)*(2*pi))/4)
#Modes
b_mode=(np.outer(b['g'],mode)*phaser).real
press_mode=(np.outer(p['g'],mode)*phaser).real
ux_mode=(np.outer(ux['g'],mode)*phaser).real
uz_mode=(np.outer(uz['g'],mode)*phaser).real


fig, axs = plt.subplots(2, 2)
ax = axs[0, 0]
c = ax.pcolor(arr_x,z,b_mode, cmap='RdBu') #buoyancy
ax.set_title('b')
fig.colorbar(c, ax=ax)
ax = axs[0, 1]
c = ax.pcolor(arr_x,z,press_mode,cmap='inferno') #pressure
ax.set_title('P')
fig.colorbar(c, ax=ax)
ax = axs[1, 0]
c = ax.pcolor(arr_x,z,ux_mode, cmap='viridis')
ax.set_title(r'$\text{u}_x$') #ux
fig.colorbar(c, ax=ax)
ax = axs[1, 1]
c = ax.pcolor(arr_x,z,uz_mode, cmap='autumn') #uz
ax.set_title(r'$\text{u}_z$')
fig.colorbar(c, ax=ax)

fig.tight_layout()
plt.savefig("/home/iiw7750/Convection/rbcheatmodeplotRa"+str(Rayleigh)+'Pr'+str(Prandtl)+'Kx'+str(kx)+".png")
plt.close()
#Eigenmodes plot
#Plotting Set-up
fig, (ax_b,ax_p,ax_x,ax_z) = plt.subplots(2,2 , figsize=(11,9), sharex=True, dpi = 500)
fig.suptitle(r'Rayleigh-Benard Modes Eigenfunctions ($\mathrm{Ra} = %.2f, \; \mathrm{Pr} = %.2f$)' %(Rayleigh, Prandtl))
#Titles
ax_b.title.set_text('Bouyancy')
ax_p.title.set_text('Pressure')
ax_x.title.set_text(r'$\text{u}_x$')
ax_z.title.set_text(r'$\text{u}_z$')
#Axes labels
    #
ax1.set_ylabel(r'$\omega$')
ax2.set_ylabel(r'$\text{f}$')
ax2.set_xlabel(r'$k_x$')

#Growth Rates
ax1.scatter(kx_global, growth_global)
plt.tight_layout()

#Mode frequency 

ax2.scatter(kx_global, freq_global)
plt.tight_layout()

#Figure Saving
plt.plot(z, b['g'].real, label='real')
plt.plot(z, b['g'].imag, label='imag')
plt.legend()
plt.savefig("/home/iiw7750/Convection/rbcmodeplot.png")
