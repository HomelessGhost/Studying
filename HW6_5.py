import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

def bin_number(n):
    return int(np.floor(np.sqrt(n)/3))


def cdf_print(x, ar):
    size = len(x)
    x_f = x
    x_f.sort()
    y = []
    for i in range(0, size):
        y.append((i+1)/size)
    plt.plot(x_f, y, ar)


def exp_statistics(sample):
    return np.exp(sample.mean())


def my_cdf(x_data, color):
    x_data = np.sort(x_data)
    len_data = len(x_data)
    for i in range(0, len_data - 1):
        x = [x_data[i], x_data[i + 1]]
        y = [i / len_data, i / len_data]
        plt.plot(x, y, color=color)


def hw_6_5():
    D = []
    B = 1
    len_sample = 20
    for i in range(10000):
        smp = np.random.uniform(0, B, len_sample)
        stat = np.max(smp)
        D.append(stat)

    xlist = mlab.frange(0, B, 0.001)
    ylist = [(x/B)**len_sample for x in xlist]

    cdf_print(D, 'blue')
    plt.plot(xlist, ylist, color='red', linestyle='--')

    D_expectancy = sum(D)/len(D)
    print(D_expectancy )
    print((1/(2**len_sample)) * 1000000)


    plt.show()

hw_6_5()