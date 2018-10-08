from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

data= np.genfromtxt(r'../Data/S2_02_003.dat', dtype = 'float')
data_len=len(data)
x=[i for i in range(data_len)]
data_1 = np.array([data[x][0] for x in range(len(data))])
data_2 = np.array([data[x][1] for x in range(len(data))])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, data_1, data_2, color = 'r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
