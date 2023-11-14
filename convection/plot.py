import matplotlib.pyplot as plt
import numpy as np
z = np.linspace(0,1,1000)
print(z)
z_bl = 0.1
Q = 1 - np.greater(z, z_bl) - np.greater(z, 1 - z_bl)
print(np.greater(z, 0.5))
plt.plot(z, Q)
plt.show()