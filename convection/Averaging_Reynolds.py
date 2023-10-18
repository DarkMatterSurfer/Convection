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

def Re_averager(filename):
    ReData = np.genfromtxt(filename, delimiter = ', ')
    Re_time = ReData[:,0]
    Re_num = ReData[:,1]

    Re_num_cut = Re_num[Re_time>=30]
    x = np.mean(Re_num_cut)
    return x 

dta_file = 'RunReynolds.csv' #input name of file you want to plot
path = '/home/brogers/reach/convection/Aelastic_approx/'

prefix_list = ['[2e6]']
Ra_list =[]
Re_list = []
for prefix in prefix_list: 
    sub_string = prefix[1:-1]
    Ra = float(sub_string)
    Ra_list.append(Ra)

for index,prefix in enumerate(prefix_list):
    file = path + '2000000.0' + dta_file
    Re_mean = Re_averager(file)
    Re_list.append(Re_mean)
    plt.scatter(Ra_list[index], Re_mean)
    x = np.log(Ra_list)
    y = np.log(Re_mean)
    # # m, b = np.polyfit(x, y, 1)
    # # plt.plot(x, b+m*x)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # plt.plot(x, slope*x + intercept, color='red', label='Regression Line')

print(type(Ra_list), type(Re_list))
combined = np.array(list(zip(Ra_list, Re_list)))
print(combined)
# plt.ylabel('Mean Reynolds Number')
# plt.xlabel('Rayleigh Number')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Simulated mean Reynolds number vs. Rayleigh number')
# plt.show()


