import numpy as np
import matplotlib.pyplot as plt
import sys 
from scipy import stats

# file_name = '10000.0RunReynolds.csv'
# ReData = np.genfromtxt(filename, delimiter = ', ')
# Re_time = ReData[:,0]
# Re_num = ReData[:,1]

# Re_num_cut = Re_num[Re_time>=125]
# x = np.mean(Re_num_cut)
# plt.scatter(x, y)
# plt.show()

def averager(filename):
    Data = np.genfromtxt(filename, delimiter = ', ')
    time = Data[:,0]
    num = Data[:,1]

    num_cut = num[time>=125]
    mean = np.mean(num_cut)
    return mean
    # print(time)
    # print(num)

dta_file_top = 'Runtopflux.csv' #input name of file you want to plot
path = '/home/brogers/reach/convection/Lx=4/FluxData_Lx=4/'

prefix_list = ['[4e4]', '[2e5]','[4e4]','[1e5]','[4e5]','[2e4]','[2e6]','[1e6]']
# , '[2e5]','[4e4]','[1e5]','[4e5]','[2e4]','[2e6]','[1e6]'
Top_list =[]
mean_flux_top = []
Bottom_list =[]
mean_flux_bottom = []
for prefix in prefix_list: 
    sub_string = prefix[1:-1]
    Ra = float(sub_string)
    Top_list.append(Ra)
    Bottom_list.append(Ra)

for index,prefix in enumerate(Top_list):
    file = path + str(prefix) + dta_file_top
    flux_mean_top = averager(file)
    mean_flux_top.append(-1*flux_mean_top)
    flux_mean_bottom = averager(file)
    mean_flux_bottom.append(-1*flux_mean_bottom)
    
    plt.scatter(Top_list, mean_flux_top)
    plt.scatter(Bottom_list, mean_flux_bottom)

    x = np.log(Top_list)
    y_top = np.log(flux_mean_top)
    # # m, b = np.polyfit(x, y, 1)
    # # plt.plot(x, b+m*x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_top)
    plt.plot(x, slope*x + intercept, color='red', label='Regression Line')
    # print(flux_mean)
    # print(averager(file))
    # print(Top_list)

    

plt.ylabel('Mean Boundary Flux')
plt.xlabel('Rayleigh Number')
plt.yscale('log')
plt.xscale('log')
plt.title('Simulated mean Boundary Flux vs. Rayleigh number')
plt.show()
print(mean_flux_bottom)
