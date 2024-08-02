

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools
import os 
from docopt import docopt
from configparser import ConfigParser
import sys
from glob import glob
path = os.path.dirname(os.path.abspath(__file__))[:-12]

if len(sys.argv) < 2:
    print('please provide config file')
    raise
else:
    configfile = sys.argv[1]
config = ConfigParser()
config.read(str(configfile))
global lx
Lx = config.getfloat('param','Lx')
Lz = config.getfloat('param','Lz')
Nz = config.getfloat('param','Nz')
ad = config.getfloat('param','back_ad')
sig = config.getfloat('param','sig')
delta = config.getfloat('param','delta')
name = config.get('param', 'name')
Ra = config.getfloat('param','Ra')
supercrit=config.getfloat('param','supercrit')
runsupcrit=config.getboolean('param','runsupcrit')
if runsupcrit == True:
    Ra = Ra *supercrit
def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    global times
    global bsensor1
    global bsensor2
    global bsensor3
    global bfixed_sensor1
    global bfixed_sensor2
    global bfixed_sensor3
    global bfixed_sensor4
    global bfixed_sensor5
    global bfixed_sensor6
    with h5py.File(filename, mode='r') as file:
        b1_dta=file['tasks']['b_1sig'][()]
        b2_dta=file['tasks']['b_2sig'][()]
        b3_dta=file['tasks']['b_3sig'][()]
        #Fixed sensors middle of bottom half
        b1fixed_dta=file['tasks']['b_1fixed'][()]
        b2fixed_dta=file['tasks']['b_2fixed'][()]
        b3fixed_dta=file['tasks']['b_3fixed'][()]
        #Fixed sensors near bottom boundary
        b4fixed_dta=file['tasks']['b_4fixed'][()]
        b5fixed_dta=file['tasks']['b_5fixed'][()]
        b6fixed_dta=file['tasks']['b_6fixed'][()]
        time_dta=file['scales']['sim_time'][()]
        for i in range(len(time_dta)):
            times.append(time_dta[i])
            #Moving sensors
            bsensor1.append(np.array(b1_dta[i][0][0]))
            bsensor2.append(np.array(b2_dta[i][0][0]))
            bsensor3.append(np.array(b3_dta[i][0][0]))
            bfixed_sensor1.append(np.array(b1fixed_dta[i][0][0]))
            bfixed_sensor2.append(np.array(b2fixed_dta[i][0][0]))
            bfixed_sensor3.append(np.array(b3fixed_dta[i][0][0]))
            bfixed_sensor4.append(np.array(b4fixed_dta[i][0][0]))
            bfixed_sensor5.append(np.array(b5fixed_dta[i][0][0]))
            bfixed_sensor6.append(np.array(b6fixed_dta[i][0][0]))
if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    global times
    global bsensor1
    global bsensor2
    global bsensor3
    global bfixed_sensor1
    global bfixed_sensor2
    global bfixed_sensor3
    global bfixed_sensor4
    global bfixed_sensor5
    global bfixed_sensor6
    #Moving senors
    sensor1 = 1/2*(Lz)-(2)*(sig)-delta#change with sig
    sensor2 = 1/2*(Lz)-(2)*(sig)-2*delta#change with sig
    sensor3 = 1/2*(Lz)-(2)*(sig)-3*delta#change with sig
        #Fixed 1
    fixedsensor1 = 1/4 + delta
    fixedsensor2 = 1/4
    fixedsensor3 = 1/4 - delta 
        #Fixed 2
    fixedsensor4 = delta
    fixedsensor5 = 2*delta
    fixedsensor6 = 3*delta
    times,bsensor1,bsensor2,bsensor3,bfixed_sensor1,bfixed_sensor2,bfixed_sensor3,bfixed_sensor4,bfixed_sensor5,bfixed_sensor6=[],[],[],[],[],[],[],[],[],[]
    
    output_path = pathlib.Path(path+"/"+name+"/buoyancy/").absolute()
    post.visit_writes(glob(path+"/"+name+"/profiles/*.h5"), main,output=output_path)
    #Plotting here line for each sensor
    power=1e1
    limbool = False
    fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
    title=r'$\sigma$='+'{}'.format(sig)+'  '+r'$\nabla_{ad}$='+'{}'.format(ad)+' '+'Ra={}'.format(Ra)+' '+'Lz={}'.format(Lz)
    fig.suptitle(title)
    if limbool:
        ax1.set_ylim(-5/(10**power),5/(10**power))
        ax2.set_ylim(-5/(10**power),5/(10**power))
        ax3.set_ylim(-5/(10**power),5/(10**power))
    ax1.set_title('Near strip',fontsize='x-small')
    ax1.plot(times,bsensor1,label='Lz={}'.format(sensor1))
    ax1.plot(times,bsensor2,label='Lz={}'.format(sensor2))
    ax1.plot(times,bsensor3,label='Lz={}'.format(sensor3))
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax2.set_title('Middle',fontsize='x-small')
    ax2.plot(times,bfixed_sensor1,label='Lz={}'.format(fixedsensor1))
    ax2.plot(times,bfixed_sensor2,label='Lz={}'.format(fixedsensor2))
    ax2.plot(times,bfixed_sensor3,label='Lz={}'.format(fixedsensor3))
    ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax3.set_title('Near boundary',fontsize='x-small')
    ax3.plot(times,bfixed_sensor4,label='Lz={}'.format(fixedsensor4))
    ax3.plot(times,bfixed_sensor5,label='Lz={}'.format(fixedsensor5))
    ax3.plot(times,bfixed_sensor6,label='Lz={}'.format(fixedsensor6))
    ax3.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    figname = 'ad{}'.format(ad)+'sig{}'.format(sig)+'ra{}'.format(Ra)+'_'+name+'bouyancyflucuationsDNS.png'
    plt.savefig(str(output_path)+figname)
    plt.close()
    sys.exit()


