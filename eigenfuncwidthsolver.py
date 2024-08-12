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
Re = config.getfloat('param','Re')
name=config.get('param','name')
kxbool = config.getboolean('param','kxbool')
#Single core printing
def print_rank(input):
    if rank == 0:
        print(str(input))
    return
def widthfinder(file):
    listZ = []
    listB = []
    halfZ = []
    halfB = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for index, row in enumerate(reader):
            # if index == 0:
            #     ad = row[6]
            #     break
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
    interwidth = 2*(Z1-1/2)
    exterwidth = 2*(Z2-1/2)
    return (interwidth,exterwidth)
sigma_list = [0.01,0.001]
reynolds_list = [0]
intrwidth_list, exterwidth_list = [],[]
for sig in sigma_list:
    for Re in reynolds_list:
        try:
            filestring = f'/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/modedata/dataraw/'+f'sig{sig}Re{Re}/'+'/*.csv'
            filelist = glob.glob(filestring)
        except:
            filestring = f'/home/brogers/Convection/eigenvalprob_plots/marginalstabilityconditions/modedata/dataraw/'+f'sig{sig}Re{Re}/'+'/*.csv'
            filelist = glob.glob(filestring)

        for file in filelist:
            ad_upper=8
            ad_lower=1
            step_factor=4
            ad_list = np.linspace(ad_lower,ad_upper,step_factor*abs(ad_upper-ad_lower)+1)
            for ad in ad_list:
                interwidth, exterwidth = widthfinder(file)
                intrwidth_list.append(interwidth)
                exterwidth_list.append(exterwidth)
                label = r'$\sigma$={}, Re={}'.format(sig,Re)
                fig, axs = plt.subplots(1, 2)
                intrax = axs[0,0]
                extrax = axs[0,1]
                intrax.settitle('Interior Limit CZ Width')
                extrax.settitle('Exterior Limit CZ Width')
                intrax.scatter(ad,intrwidth_list,label = label)
                extrax.scatter(ad,exterwidth_list,label = label)

folderstring= "sig{}Re{}Nz{}".format(sig,Re,Nz)
try: 
    full_dir = '/home/iiw7750/Convection/eigenvalprob_plots/marginalstabilityconditions/' + 'sig{}/'.format(sig)+'modeplots/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    plt.savefig(full_dir+"rbcconvectivewidth"+folderstring+".png")
except:
    full_dir = '/home/brogers/Convection/eigenvalprob_plots/marginalstabilityconditions/' + 'sig{}/'.format(sig)+'modeplots/'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    plt.savefig(full_dir+"rbcconvectivewidth"+folderstring+".png")
print(full_dir+"rbcconvectivewidth"+folderstring+".png")
full_dir = "/home/iiw7750/Convection/"+name 
full_dir = path+'/eigenvalprob_plots/marginalstabilityconditions/'+'sig{}/'.format(sig)+'modeplots/'
if not os.path.exists(full_dir):
    os.makedirs(full_dir)
plt.savefig(full_dir+"rbcconvectivewidth"+folderstring+".png")
print(full_dir+"rbcconvectivewidth"+folderstring+".png")
plt.close()