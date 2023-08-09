import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import csv
#plt.style.use('ggplot')
import sys

#Reading file -> conversion into dataframe
dta_file = 'Reynolds1.csv' #input name of file you want to plot
path = '/home/brogers/reach/convection/ReynoldsData1'

prefix_list = ['[2e4]', '[4e4]', '[1e5]', '[2e5]', '[4e5]', '[1e6]', '[2e6]']

for index, prefix in enumerate(prefix_list):
    ReynoldsList = np.genfromtxt(path + prefix + dta_file, delimiter = ', ')
    # ReynoldsList.rename( columns={0 :'Articles'}, inplace=True )
    # colnames = ['Time','Re']
    # frame = pd.DataFrame(ReynoldsList, columns = colnames)
    time = ReynoldsList[:, 0]
    Re = ReynoldsList[:, 1]
    # print(Re)
    # sys.exit()

    #Accessing Time and Reynolds Number

    # time = ReynoldsList['Time']
    # Re = ReynoldsList['Re']

    #Plotting Data
    plt.plot(time, Re, label='Ra='+prefix)


plt.title('Simulated Maximum Reynolds Number in Time-domain')
plt.xlabel('Time')
plt.legend()
plt.ylabel('Reynolds[Re]')
plt.show() 
greater = np.where(time)