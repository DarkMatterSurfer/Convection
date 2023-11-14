import matplotlib.pyplot as plt
import numpy as np

kappa = 1 
Tx = np.linspace(0,1,50) #temperature 

sig = 30 
e = 0.1
Tbump = 0.5
Tplus = Tx -Tbump + e
Tminus = Tx -Tbump - e
A = 0.5
pi = np.pi

koopa = kappa*A*(((-pi/2)+np.arctan(sig*Tplus*Tminus))/((pi/2)+np.arctan(sig*e*e)))
#Plotting Koopa
plt.plot(Tx.flatten(), koopa)
plt.xlabel('Temperature')
plt.ylabel('Radiative Conductivity')
plt.show()
