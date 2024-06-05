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

meanNuss1e6 = [4.245208055	,4.323141722	,3.495836751	,4.204572801,	4.644736469,	5.368421985,	4.447148319]
meanNuss2e6 = [5.385896869	,5.609825286,	5.326904591,	5.959108636,	4.93109487,	5.970119451,	4.628803322]
meanNuss4e6 = [6.353129557	,5.342056941	,6.300756625	,6.591442453	,5.678098703	,5.890881464	,5.952970844]
meanNuss1e7 = [8.260645308	,8.591924604	,7.978619452	,5.99938351	,8.273839429	,7.175487156	,7.753026138]
meanNuss2e7 = [8.310815954	,8.094504042	,8.310461078	,9.587334976	,8.097637779	,9.144887157	,8.438790086]
meanNuss4e7 = [9.155097703,	9.725867619,	11.75246534,8.840253032,	9.766538995,	9.548487425,	9.242469968,10.73199684,	11.51601422]

meanReyn1e6 = [1113.910509	,1094.533531	,1116.938475	,1134.778129	,1120.098642	,1125.360987	,1140.271693]
meanReyn2e6 = [1616.139211	,1593.109591	,1619.258551	,1604.182882	,1584.793442	,1614.508768	,1637.808135]
meanReyn4e6 = [2321.781474	,2300.971775	,2332.455948	,2344.042585	,2339.619311	,2355.994494	,2371.694136]
meanReyn1e7 = [3651.105934	,3706.451822	,3705.500416	,3707.907164	,3704.064088	,3672.244153	,3747.773984]
meanReyn2e7 = [5599.975723	,5642.481816	,5350.273212	,5502.208157	,5552.395673	,5579.978723	,5593.446565]
meanReyn4e7 = [7613.345048	,7692.366397	,7723.597606,7782.369981,7713.123116	,8079.311849	,7866.391249	,7782.369981,7763.144006]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,10), sharex=True, dpi = 500)
fig.suptitle('Globally Averaged Fluid Quantities')
ax1.set_ylabel("Nu")
ax2.set_ylabel("Re")
def ampplotter(amps,mean,mark,axis):
    axis.scatter(amps,mean,label = mark)
#Nusselt Plot ax1
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanNuss1e6, "Ra = 1e6", ax1)
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanNuss2e6, "Ra = 2e6", ax1)
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanNuss4e6, "Ra = 4e6", ax1)
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanNuss1e7, "Ra = 1e7", ax1)
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanNuss2e7, "Ra = 2e7", ax1)
ampplotter([0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.85,0.9],meanNuss4e7, "Ra = 4e7", ax1)
#Reynolds Plot ax2
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanReyn1e6, "Ra = 1e6", ax2)
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanReyn2e6, "Ra = 2e6", ax2)
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanReyn4e6, "Ra = 4e6", ax2)
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanReyn1e7, "Ra = 1e7", ax2)
ampplotter([0.3,0.4,0.5,0.6,0.7,0.8,0.9],meanReyn2e7, "Ra = 2e7", ax2)
ampplotter([0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.85,0.9],meanReyn4e7, "Ra = 4e7", ax2)

ax1.legend(loc = 'lower right', prop = {'size':7})
ax2.legend(loc = 'upper right', prop = {'size':7})
ax2.set_xlabel("Amplitude")
ax1.set_ylim(0,14)
ax2.set_ylim(500,12000)
plt.savefig(path+"/globalaverageplot.png")

