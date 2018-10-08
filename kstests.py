import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy.stats import kstest
from scipy.stats import ks_2samp



data = np.genfromtxt('../Data/S2_03_001.dat', dtype = 'float')
len_data = len(data)
#data = list(map(abs, data))
data.sort()

for i in range(0, len_data-1):
    x = [data[i], data[i+1]]
    y = [i/len_data, i/len_data]
    plt.plot(x, y, color = 'b')


def magic(x):
#    return 1 / x
    return 1 / (x**2)


#xmin = data[0]
#xmax = data[len_data-1]

#dx = 0.001
#xlist = mlab.frange(xmin, xmax, dx)
#ylist = [magic(x) for x in xlist]
#plt.plot(xlist, ylist, color = 'r')
plt.title('Выборка')
#plt.yscale('log')
#plt.xscale('log')
plt.show()
