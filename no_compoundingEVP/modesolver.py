import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import time
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD
import os
import sys
from Convection.no_compoundingEVP.EVP_methods import modesolver
path = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 2:
    # raise
    try:
        configfile = path + "/options.cfg"
    except:
        print('please provide config file')
        raise
else:
    configfile = sys.argv[1]
from configparser import ConfigParser
config = ConfigParser()
config.read(str(configfile))
# Parameters
Nz = config.getfloat('param', 'Nz')
Nx = config.getfloat('param','Nx')
Rayleigh = config.getfloat('param', 'Ra') 
Prandtl = config.getfloat('param', 'Pr')
L_x = config.getfloat('param','Lx')
Lz = config.getfloat('param','Lz')
pi=np.pi
kx = config.getfloat('param','kx')
kx_global = eval(config.get('param','kx_global'))
wavenum_list = []
NEV = config.getint('param','NEV')
A = config.getfloat('param','A')
sig = sig_og = config.getfloat('param','sig')
ad = config.getfloat('param', 'adiabat_mean')  
solver = modesolver (Rayleigh, Prandtl, kx, Nz, A, ad, sig,Lz,NEV=10, target=0)
name=config.get('param','name')

# Bases
zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord, dtype=np.complex128)
zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, Lz))
z = dist.local_grid(zbasis)
arr_x = np.linspace(0,L_x,Nx)
mode=np.exp(1j*kx*arr_x)
sp = solver.subproblems[0]
evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
evals = evals[np.argsort(-evals.real)]
print(f"Slowest decaying mode: λ = {evals[0]}")
solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0])

#Fields
b = solver.state[1]
ux = solver.state[2]
p = solver.state[0]
uz = solver.state[3]
b.change_scales(1)
p.change_scales(1)
ux.change_scales(1)
uz.change_scales(1)
#Heat Map
pi=np.pi
phase=1
phaser=np.exp(((1j*phase)*(2*pi))/4)
#Modes
# b['g']=b['g']-(2 * (z - 1/2))
b_mode=(np.outer(b['g'],mode)*phaser).real
b_mode = b_mode-2*(z[..., np.newaxis]-1/2)
press_mode=(np.outer(p['g'],mode)*phaser).real
ux_mode=(np.outer(ux['g'],mode)*phaser).real
uz_mode=(np.outer(uz['g'],mode)*phaser).real

modeslist = [b_mode,ux_mode,uz_mode]
full_dir = path+"/"+name +'/'
if not os.path.exists(full_dir):
    os.makedirs(full_dir)
np.save(full_dir+'modedata.npy', np.array(modeslist, dtype=object),allow_pickle=True)
# sys.exit()
fig, axs = plt.subplots(2, 2)
ax = axs[0, 0]
ax.set_aspect('equal')
ax.set_adjustable('box', share=True)
c = ax.pcolor(arr_x,z,b_mode, cmap='RdBu') #buoyancy
ax.set_title('b')
fig.colorbar(c, ax=ax)

ax = axs[0, 1]
ax.set_aspect('equal')
ax.set_adjustable('box', share=True)
c = ax.pcolor(arr_x,z,press_mode,cmap='inferno') #pressure
ax.set_title('P')
fig.colorbar(c, ax=ax)
ax = axs[1, 0]
ax.set_aspect('equal')
ax.set_adjustable('box', share=True)
c = ax.pcolor(arr_x,z,ux_mode, cmap='viridis')
ax.set_title(r'$\text{u}_x$') #ux
fig.colorbar(c, ax=ax)

ax = axs[1, 1]
ax.set_aspect('equal')
ax.set_adjustable('box', share=True)
c = ax.pcolor(arr_x,z,uz_mode, cmap='autumn') #uz
ax.set_title(r'$\text{u}_z$')
fig.colorbar(c, ax=ax)

folderstring= "Ra"+str(Rayleigh)+"Pr"+str(Prandtl)
full_dir = path+"/eigenvalprob_plots/"+folderstring
if not os.path.exists(full_dir):
    os.makedirs(full_dir)
plt.savefig(full_dir+"/"+name+"rbcheatmodeplotRa"+str(Rayleigh)+'Pr'+str(Prandtl)+'Kx'+str(kx)+".png")
full_dir = path+"/"+name 
plt.savefig(full_dir+"/rbcheatmodeplotRa"+str(Rayleigh)+'Pr'+str(Prandtl)+'Kx'+str(kx)+".png")
plt.close()

#Eigenmodes plot
#Plotting Set-up
fig, ((ax_b,ax_p),(ax_x,ax_z)) = plt.subplots(2,2, figsize=(11,9),  dpi = 500)
fig.suptitle(r'Rayleigh-Benard Modes Eigenfunctions ($\mathrm{Ra} = %.2f, \; \mathrm{Pr} = %.2f$)' %(Rayleigh, Prandtl))
#Titles
ax_b.title.set_text('Bouyancy')
ax_p.title.set_text('Pressure')
ax_x.title.set_text(r'$\text{u}_x$')
ax_z.title.set_text(r'$\text{u}_z$')
#Axes labels
    #xlabels
ax_b.set_xlabel('z')
ax_p.set_xlabel('z')
ax_z.set_xlabel('z')
ax_x.set_xlabel('z')
    #ylabels
ax_b.set_ylabel('b')
ax_p.set_ylabel('P')
ax_x.set_ylabel(r'$\text{u}_x$')
ax_z.set_ylabel(r'$\text{u}_z$')
#Buoyancy
ax_b.plot(z, b['g'].real)
ax_b.plot(z, b['g'].imag)
plt.tight_layout()
#Pressure
ax_p.plot(z, p['g'].real)
ax_p.plot(z, p['g'].imag)
plt.tight_layout()
#U_x
ax_x.plot(z, ux['g'].real)
ax_x.plot(z, ux['g'].imag)
plt.tight_layout()
#u_z
ax_z.plot(z, uz['g'].real)
ax_z.plot(z, uz['g'].imag)
plt.tight_layout()

#Figure Saving
full_dir = path+"/"+name 
plt.savefig(full_dir+"/rbc1DmodeplotRa"+str(Rayleigh)+'Pr'+str(Prandtl)+'Kx'+str(kx)+".png")
plt.close()