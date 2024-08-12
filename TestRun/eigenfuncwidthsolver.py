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
import csv
from EVP_methods_CHEBBED import modesolver
import glob 
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
#listRa = [2735.941162109375, 3577.5636875629425, 5165.456945372862, 9274.113861394537, 44323.74339615709, 9687205.171682963, 49703687.47267801, 123312712.91439304, 228170666.79154903, 364538291.86618567, 531944670.52836716, 730060294.2810587, 960253866.0740244, 1217665669.0401568, 1505285672.6525476, 1826114381.692019, 2177204488.0353947, 2555132317.8677106, 2970590844.1616306, 3407911713.065308, 3883405254.085041, 4378311880.704277, 4917049475.400311, 5473458760.519464, 6074790899.736689, 6688795643.997185, 7350163377.351398]
#listNab = [1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5]
#wavenum_list = [6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 12.566370614359172, 25.132741228718345, 37.69911184307752, 43.982297150257104, 50.26548245743669, 56.548667764616276, 62.83185307179586, 69.11503837897544, 69.11503837897544, 75.39822368615503, 81.68140899333463, 81.68140899333463, 87.96459430051421, 87.96459430051421, 94.24777960769379, 100.53096491487338, 100.53096491487338, 106.81415022205297, 106.81415022205297, 113.09733552923255, 113.09733552923255, 113.09733552923255]
listRa = [9687205.171682963]
listNab = [2.25]
wavenum_list = [12.566370614359172]

supercrit=config.getfloat('param','supercrit')
runsupcrit=config.getboolean('param','runsupcrit')
if runsupcrit == True:
    ra_sig = listRa *supercrit
else:
    supercrit = 1
    ra_sig = listRa
Prandtl = config.getfloat('param', 'Pr')
Re = config.getfloat('param','Re')
Lx = config.getfloat('param','Lx')
Lz = config.getfloat('param','Lz')
pi=np.pi
# kx = config.getfloat('param','kx')
NEV = config.getfloat('param','NEV')
target = config.getfloat('param','target')
sig = sig_og = config.getfloat('param','sig')
modecompbool = config.getboolean('param','modecompbool')
name=config.get('param','name')
zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord, dtype=np.complex128)
zbasis = d3.ChebyshevT(zcoord, size=Nz, bounds=(0, Lz))
z = dist.local_grid(zbasis)
arr_x = np.linspace(0,Lx,Nx)
pi=np.pi
phase=1
phaser=np.exp(((1j*phase)*(2*pi))/4)
kxbool = config.getboolean('param','kxbool')
#Single core printing
def print_rank(input):
    if rank == 0:
        print(str(input))
    return
# Bases
def modewrapper (Rayleigh, Prandtl, kx, Nz, ad, sig,Lz,Re):
    solver = modesolver(Rayleigh, Prandtl, kx, Nz, ad, sig,Lz,Re)
    sp = solver.subproblems[0]
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    evals = evals[np.argsort(-evals.imag)]
    mode=np.exp(1j*kx*arr_x)
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
    # b_c.require_gridW
    p = np.concatenate((p_c['g'][()][0], p_r['g'][()][0]),axis = 0)
    b = np.concatenate((b_c['g'][()][0], b_r['g'][()][0]),axis = 0)
    ux = np.concatenate((ux_c['g'][()][0], ux_r['g'][()][0]),axis = 0)
    uz = np.concatenate((uz_c['g'][()][0], uz_r['g'][()][0]),axis = 0)


    return (b,p,ux,uz,mode)
def widthfinder(file):
    listZ = []
    listB = []
    halfZ = []
    halfB = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for index, row in enumerate(reader):
            if not index == 0:
                listZ.append(eval(row[1]).real)
                listB.append(eval(row[2]).real)
    halfZ = listZ[round(Nz/2):]
    halfB = listB[round(Nz/2):]
    firstRoot = False
    for index, buoy in enumerate(halfB):
        if not(buoy>0) and not (firstRoot):
            indx0 = index 
            firstRoot = True
        if buoy>0 and (firstRoot):
            indx1 = index
            break
    if not(firstRoot):
        raise
    z0 = halfZ[indx0-1]
    b0 = halfB[indx0-1]
    z1 = halfZ[indx0]
    b1 = halfB[indx0]
    Z1 = z0 - (b0*(z1-z0)/(b1-b0))
    z0 = halfZ[indx1-1]
    b0 = halfB[indx1-1]
    z1 = halfZ[indx1]
    b1 = halfB[indx1]
    Z2 = z0 - (b0*(z1-z0)/(b1-b0))
    print('Smaller width:',2*(Z1-1/2))
    print('Larger Width:',2*(Z2-1/2))
    plt.plot(halfZ,halfB)
    plt.axvline(Z1)
    plt.axvline(Z2)
    plt.axhline(0)
    plt.savefig(path+f'/{name}/debug.png')
    print(path+f'/{name}/debug.png')
    plt.close()

filelist = glob.glob('/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/dataraw/*.csv')
for file in filelist:
    print(filelist)