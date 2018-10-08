from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
from matplotlib import mlab


data= np.genfromtxt(r'text1.txt', dtype = 'float')
data_len=len(data)
x=[i for i in range(data_len)]
data_1 = np.array([data[x][0] for x in range(len(data))])
data_2 = np.array([data[x][1] for x in range(len(data))])
data_2 = list(map(lambda x : math.log(x), data_2))
for i in data_2:
    print(i)
plt. plot(data_1, data_2, color='b')
plt.scatter(data_1, data_2, color='r')
#plt.yscale('log')
#plt.xscale('log')



plt.show()
