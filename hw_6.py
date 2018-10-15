#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math
import scipy.special as sc
import scipy.stats as st
import scipy.integrate as integ
import matplotlib.pyplot as plt


def my_statistic(samp):
    return math.exp(samp.mean())


def se_boot_eval(samp):
    samp = np.array(samp)
    avg = samp.mean()
    size = samp.size
    n_samp = []
    for i in range(size):
        n_samp.append(samp[i]-avg)
    n_samp = np.array(n_samp)
    return n_samp.mean()


def cdf_print(x, ar):
    size = len(x)
    x_f = x
    x_f.sort()
    xf = reversed(x_f)
    y = []
    for i in range(0, size):
        y.append((i+1)/size)
    plt.plot(x_f, y, ar)


def hist_width(n):
    return 1+math.floor(math.log(n, 2))


def find_nearest(array, value):
    array = np.asarray(array)
    array = np.sort(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest_inv(array, value):
    Y = [i/len(array) for i in range(len(array))] 
    return find_nearest(Y, value)


def num_bin(bins, val):
    for i in range(1, len(bins)):
        if val < bins[i]:
            return i


def rand_samp_in_bin(bins, samp, n_b):
    a = bins[n_b-1]
    b = bins[n_b]
    vals = []
    for i in samp:
        if i >= a and i < b:
            vals.append(i)
    if len(vals) == 0:
        return samp[np.random.randint(0, len(samp))]
    return vals[np.random.randint(0, len(vals))]


def stat_give_b(r_x, val_b):
    n, bins, _ = plt.hist(r_x, val_b)
    n_normed = [n[i]/size for i in range(len(n))]
    T_b_n = []
    for i in range(boot_iter):
        n_r_x = []
        for j in range(size):
            val = np.random.uniform(bins[0], bins[len(bins)-1])
            n_b = num_bin(bins, val)
            n_r_x.append(rand_samp_in_bin(bins, r_x, n_b))
        n_r_x = np.array(n_r_x)
        T_b_n.append(my_statistic(n_r_x))
    return T_b_n


size = 100
boot_iter = 1000

T_b = []
for i in range(boot_iter):
    r_x = np.random.normal(5, 1, size)
    T_b.append(my_statistic(r_x))
r_x=np.random.normal(5, 1, size)
T_b_n=stat_give_b(r_x, 70)
cdf_print(T_b_n,"r.")
plt.hist(n_r_x, hist_width(n_size))
cdf_print(n_r_x,"b.")
cdf_print(r_x,"b.")
print(se_boot_eval(T_b_n))
cdf_print(T_b,"b.")
plt.hist(T_b, hist_width(boot_iter))
T_b_n[find_nearest_inv(T_b_n, 0.975)]
T_b_n[find_nearest_inv(T_b_n, 0.025)]
plt.hist(T_b_n, hist_width(boot_iter))
math.exp(5)


def my_statistic(samp):
    return samp.max()


size=50
r_x_2=np.random.rand(size)
T_b_n_2=stat_give_b(r_x_2, 25)
T_b_2=[]
for i in range(boot_iter):
    r_x_2=np.random.rand(size)
    T_b_2.append(my_statistic(r_x_2))

cdf_print(T_b_2, "r.")
cdf_print(T_b_n_2, "b.")
plt.hist(T_b_n_2, hist_width(boot_iter))
plt.hist(T_b_2, hist_width(boot_iter))




