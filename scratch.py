import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy.stats import kstest
from scipy.stats import ks_2samp
import math



x_data = np.genfromtxt('../Data/S2_04_002.dat', dtype ='float')
len_data = len(x_data)
#data = list(map(abs, data))
x_data.sort()
y_data = [1 - i/len_data for i in range(0, len_data)]



for i in range(0, len_data - 1):
    x = [x_data[i], x_data[i + 1]]
    y = [1 - i/len_data, 1 - i/len_data]
    plt.plot(x, y, color = 'b')

xmin = x_data[0]
xmax = x_data[len_data - 1]

dx = 0.01
x_list = mlab.frange(xmin, xmax, dx)
y_list = list(map(lambda x: 1 / (x**3), x_list))


plt.plot(x_list, y_list, color ='r')
plt.title('4-е задание. Выборка №2. Момент 1')



plt.yscale('log')
plt.xscale('log')
plt.show()
