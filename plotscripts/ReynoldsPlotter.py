import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import csv
#plt.style.use('ggplot')
import sys

#Reading file -> conversion into dataframe
dta_file = 'RunReynolds.csv' #input name of file you want to plot
path = '/home/brogers/reach/convection/'

prefix_list = [6e6]
for index, prefix in enumerate(prefix_list):
    prefix_tostring = str(prefix)
    ReynoldsList1 = np.genfromtxt(path + prefix_tostring + dta_file, delimiter = ', ')
    ReynoldsList2 = np.genfromtxt('/home/brogers/reach/convection/Bous_approx/Lx=4/Pr=1/ReynoldsData1/' + prefix_tostring + dta_file, delimiter = ', ')
    # ReynoldsList.rename( columns={0 :'Articles'}, inplace=True )
    # colnames = ['Time','Re']
    # frame = pd.DataFrame(ReynoldsList, columns = colnames)
    time1 = ReynoldsList1[:, 0]
    Re1 = ReynoldsList1[:, 1]

    time2 = ReynoldsList2[:, 0]
    Re2 = ReynoldsList2[:, 1]
    # print(Re)
    # sys.exit()

    #Accessing Time and Reynolds Number

    # time = ReynoldsList['Time']
    # Re = ReynoldsList['Re']

    #Plotting Data
    plt.plot(time1, Re1, label='No Bump')
    plt.plot(time2, Re2, label='Conductivity Bump')


plt.title('Simulated Maximum Reynolds Number in Time-domain')
plt.xlabel('Time')
plt.legend()
plt.ylabel('Reynolds[Re]')
plt.show() 
