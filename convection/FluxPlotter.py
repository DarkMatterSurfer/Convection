import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import csv
#plt.style.use('ggplot')
import sys

#Reading file -> conversion into dataframe
dta_filetop = 'Runtopflux.csv' #input name of file you want to plot
dta_filebottom = 'Runbottomflux.csv'
path = '/home/brogers/reach/convection/Lx=16/'

prefix_list = ['[2e4]', '[4e4]', '[1e5]', '[2e5]', '[4e5]', '[1e6]', '[2e6]']
pref_list = [] #List of float converted rayleigh numbers

for prefix in prefix_list: 
    sub_string = prefix[1:-1]
    Ra = float(sub_string)
    pref_list.append(Ra)
    
for index, prefix in enumerate(pref_list):
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



    plt.title('Simulated Heat Flux at Boundaries in Time-domain' + '\n' + "Ra=" + str(prefix))
    plt.xlabel('Time')
    plt.legend()
    plt.ylabel('Heat Flux')
    plt.show() 