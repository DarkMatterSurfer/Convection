import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import csv
#plt.style.use('ggplot')
import sys

#Reading file -> conversion into dataframe
dta_filetop = 'Runtopflux.csv' #input name of file you want to plot
dta_filebottom = 'Runbottomflux.csv'
path = '/home/brogers/reach/convection/'
Rayleigh = 4e5
prefix_list = [str(Rayleigh)]


for index, prefix in enumerate(prefix_list):
    TopFluxList = np.genfromtxt(path + prefix + dta_filetop, delimiter = ', ')
    BottomFluxList = np.genfromtxt(path + prefix + dta_filebottom, delimiter = ', ')
    # ReynoldsList.rename( columns={0 :'Articles'}, inplace=True )
    # colnames = ['Time','Re']
    # frame = pd.DataFrame(ReynoldsList, columns = colnames)
    time_tflux= TopFluxList[:, 0]
    topflux = TopFluxList[:, 1]

    time_bflux = BottomFluxList[:,0]
    bottomflux = BottomFluxList[:,1]
    # print(Re)
    # sys.exit()

    #Accessing Time and Reynolds Number

    # time = ReynoldsList['Time']
    # Re = ReynoldsList['Re']

    #Plotting Data
    plt.plot(time_tflux, topflux, label='Top Boundary FLux')
    plt.plot(time_bflux, bottomflux, label='Bottom Boundary FLux')



plt.title('Simulated Heat Flux at Boundaries in Time-domain' + '\n' + "Ra=" + str(Rayleigh))
plt.xlabel('Time')
plt.legend()
plt.ylabel('Heat Flux')
plt.show() 