import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import time
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import os
import sys
from EVP_methods_CHEBBED import modesolver
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
Nz = config.getint('param', 'Nz')
Nx = config.getint('param','Nx')
Rayleigh = config.getfloat('param', 'Ra')
supercrit=config.getfloat('param','supercrit')
runsupcrit=config.getboolean('param','runsupcrit')
if runsupcrit == True:
    ra_sig = Rayleigh *supercrit
else:
    supercrit = 1
    ra_sig = Rayleigh 
Prandtl = config.getfloat('param', 'Pr')
Re = config.getfloat('param','Re')
Lx = config.getfloat('param','Lx')
Lz = config.getfloat('param','Lz')
pi=np.pi
kx = config.getfloat('param','kx')
wavenum_list = []
NEV = config.getfloat('param','NEV')
target = config.getfloat('param','target')
ad = config.getfloat('param','back_ad')
sig = sig_og = config.getfloat('param','sig')
solver = modesolver(Rayleigh, Prandtl, kx, Nz, ad, sig,Lz,Re)
modecompbool = config.getboolean('param','modecompbool')
name=config.get('param','name')
#Single core printing
def print_rank(input):
    if rank == 0:
        print(str(input))
    return
# Bases
zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord, dtype=np.complex128)
zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, Lz))
z = dist.local_grid(zbasis)
z_match = Lz/2
arr_x = np.linspace(0,Lx,Nx)
mode=np.exp(1j*kx*arr_x)
sp = solver.subproblems[0]
evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
evals = evals[np.argsort(-evals.imag)]
print(f"Slowest decaying mode: λ = {evals[0]}")
solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0])

#Fields
    #Compounded descreetized state fields
p_r = solver.state[0]
b_r = solver.state[1]
ux_r = solver.state[2]
uz_r = solver.state[3]
p_c = solver.state[14]
b_c = solver.state[15]
ux_c = solver.state[16]
uz_c = solver.state[17]
b_r.change_scales(1)
p_r.change_scales(1)
ux_r.change_scales(1)
uz_r.change_scales(1)
b_c.change_scales(1)
p_c.change_scales(1)
ux_c.change_scales(1)
uz_c.change_scales(1)
    #Combined state field
# b_c.require_grid
p = np.concatenate((p_c['g'][()][0], p_r['g'][()][0]),axis = 0)
b = np.concatenate((b_c['g'][()][0], b_r['g'][()][0]),axis = 0)
ux = np.concatenate((ux_c['g'][()][0], ux_r['g'][()][0]),axis = 0)
uz = np.concatenate((uz_c['g'][()][0], uz_r['g'][()][0]),axis = 0)
#Heat Map
pi=np.pi
phase=1
phaser=np.exp(((1j*phase)*(2*pi))/4)
#Modes
# b['g']=b['g']-(2 * (z - 1/2))
b_mode=(np.outer(b,mode)*phaser).real
press_mode=(np.outer(p,mode)*phaser).real
ux_mode=(np.outer(ux,mode)*phaser).real
uz_mode=(np.outer(uz,mode)*phaser).real

modeslist = [b_mode,ux_mode,uz_mode]
if modecompbool:
    full_dir = path+"/"+name +'/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    np.save(full_dir+'/'+name+'modedata.npy', np.array(modeslist, dtype=object),allow_pickle=True)

#Heat map plotting
fig, axs = plt.subplots(2, 2)
fig.suptitle('RBC Heatmap Modes: '+r'Ra= '+'{}'.format(Rayleigh)+r'$S_{crit}=$'+str(supercrit)+r' $\nabla$= '+'{}'.format(ad)+r' $\sigma$= '+'{}'.format(sig)+r' $k_{x}$= '+'{}'.format(kx))
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

folderstring= "ad{}sig{}Ra{}sprcrt{}Re{}_kx{}Nz{}".format(ad,sig,Rayleigh,supercrit,Re,kx,Nz)
try: 
    full_dir = '/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/' + 'sig{}/'.format(sig)+'modeplots/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    plt.savefig(full_dir+"rbcheatmodeplot"+folderstring+".png")
except:
    full_dir = '/home/brogers/Convection/eigenvalprob_plots/marginalstabilityconditions/' + 'sig{}/'.format(sig)+'modeplots/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    plt.savefig(full_dir+"rbcheatmodeplot"+folderstring+".png")
print(full_dir+"rbcheatmodeplot"+folderstring+".png")
full_dir = "/home/iiw7750/Convection/"+name 
full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'sig{}/'.format(sig)+'modeplots/'
if not os.path.exists(full_dir):
    os.makedirs(full_dir)
plt.savefig(full_dir+"rbcheatmodeplot"+folderstring+".png")
print(full_dir+"rbcheatmodeplot"+folderstring+".png")
plt.close()


#Eigenmodes plot
#Plotting Set-up
fig, ((ax_b,ax_p),(ax_x,ax_z)) = plt.subplots(2,2, figsize=(11,9),  dpi = 500)
fig.suptitle('RBC Eigenfunctions: '+r'Ra= '+'{}'.format(Rayleigh)+r' $\nabla$= '+'{}'.format(ad)+r' $\sigma$= '+'{}'.format(sig)+r' $k_{x}$= '+'{}'.format(kx))
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
ax_b.plot(z, b.real,label='Re')
ax_b.plot(z, b.imag,label='Im')
plt.tight_layout()
#Pressure
ax_p.plot(z, p.real,label='Re')
ax_p.plot(z, p.imag,label='Im')

#U_x
ax_x.plot(z, ux.real,label='Re')
ax_x.plot(z, ux.imag,label='Im')

#u_z
ax_z.plot(z, uz.real,label='Re')
ax_z.plot(z, uz.imag,label='Im')
plt.legend()

#Figure Saving
folderstring= "ad{}sig{}Ra{}sprcrt{}Re{}_kx{}Nz{}".format(ad,sig,Rayleigh,supercrit,Re,kx,Nz)
try: 
    full_dir = '/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/' + 'sig{}/'.format(sig)+'modeplots/eigenfunc/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    plt.savefig(full_dir+"rbcheatmodeplot"+folderstring+".png")
except:
    full_dir = '/home/brogers/Convection/eigenvalprob_plots/marginalstabilityconditions/' + 'sig{}/'.format(sig)+'modeplots/eigenfunc/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    plt.savefig(full_dir+"rbeigenfuncplot"+folderstring+".png")
print(full_dir+"rbeigenfuncplot"+folderstring+".png")
full_dir = "/home/iiw7750/Convection/"+name 
full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'sig{}/'.format(sig)+'modeplots/eigenfunc/'
if not os.path.exists(full_dir):
    os.makedirs(full_dir)
plt.savefig(full_dir+"rbeigenfuncplot"+folderstring+".png")
print(full_dir+"rbeigenfuncplot"+folderstring+".png")
plt.close()