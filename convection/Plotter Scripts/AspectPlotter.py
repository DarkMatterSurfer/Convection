import numpy as np
import matplotlib.pyplot as plt

# # Lx=4 Pr=1
# path_Lx4Pr1='/home/brogers/reach/convection/Lx=4/Pr=1/'
# Lx4Pr1 = np.array(np.genfromtxt(path_Lx4Pr1 + 'AspectRatio.csv', delimiter = ', '))
# # Lx=4 Pr=10
# path_Lx4Pr10='/home/brogers/reach/convection/Lx=4/Pr=10/'
# Lx4Pr10 = np.array(np.genfromtxt(path_Lx4Pr10 + 'AspectRatio.csv', delimiter = ', '))
# Lx=16 Pr=1
path_Lx16Pr1='/home/brogers/reach/convection/Lx=16/Pr=1/'
Lx16Pr1 = np.array(np.genfromtxt(path_Lx16Pr1 + 'AspectRatio.csv', delimiter = ', '))
# # Lx=16 Pr=10
# path_Lx16Pr10='/home/brogers/reach/convection/Lx=16/Pr=10/'
# Lx16Pr10 = np.array(np.genfromtxt(path_Lx16Pr1 + 'AspectRatio.csv', delimiter = ', '))

#Lx=16 Pr=1 Intialized at 6e4
path_Lx16Pr1Hy='/home/brogers/reach/convection/Lx=16/HysterRun/'
Lx16Pr1Hy = np.array(np.genfromtxt(path_Lx16Pr1Hy + 'AspectRatio.csv', delimiter = ', '))

###Plotting Script

#Plot Lx=4 Pr=1 Aspect Ratio
# Ra_Lx4Pr1 = Lx4Pr1[:,0]
# Aspect_Lx4Pr1 = Lx4Pr1[:,1]
# plt.scatter(Ra_Lx4Pr1, Aspect_Lx4Pr1, label='Lx=4 Pr=1')
# #Plot Lx=4 Pr=10 Aspect Ratio
# Ra_Lx4Pr10 = Lx4Pr10[:,0]
# Aspect_Lx4Pr10 = Lx4Pr10[:,1]
# plt.scatter(Ra_Lx4Pr10, Aspect_Lx4Pr10, label='Lx=4 Pr=10')
#Plot Lx=16 Pr=1 Aspect Ratio
Ra_Lx16Pr1 = Lx16Pr1[2:,0]
Aspect_Lx16Pr1 = Lx16Pr1[2:,1]
plt.scatter(Ra_Lx16Pr1, Aspect_Lx16Pr1, s = 30, label = 'Lx=16 Pr=1')
# Ra_Lx16Pr10 = Lx16Pr10[2:,0]
# Aspect_Lx16Pr10 = Lx16Pr10[2:,1]
# plt.scatter(Ra_Lx16Pr10, Aspect_Lx16Pr10, label= 'Lx=16 Pr=10')

#Plot Lx=4 Pr=1 Intialized=6e4 Aspect Ratio
Ra_Lx16Pr1Hy = Lx16Pr1Hy[2:,0]
Aspect_Lx16Pr1Hy = Lx16Pr1Hy[2:,1]
plt.scatter(Ra_Lx16Pr1Hy, Aspect_Lx16Pr1Hy, s = 30, label = 'Lx=16 Pr=1 Intialized=6e4', color='green')

### Figure Settings

plt.xlabel('Rayleigh Number', fontsize = 15)
plt.xscale('log')
plt.ylabel('Aspect Ratio', fontsize = 15)
plt.title('Simulated Aspect Ratios [Lx=16 Pr=1]')
plt.legend()
plt.show()