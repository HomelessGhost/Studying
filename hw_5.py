#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import scipy.special as sc
import scipy.stats as st
import scipy.integrate as integ
import matplotlib.pyplot as plt
from scipy import stats


# In[60]:


def count_pos(samp1, samp2):
    sum = 0
    for i in range(0, len(samp1)):
        if samp1[i] > samp2[i]:
            sum+=1
    return sum


def count_neg(samp1, samp2):
    sum = 0
    for i in range(0, len(samp1)):
        if samp1[i] < samp2[i]:
            sum += 1
    return sum


def sign_test_2samp(samp1, samp2, alt='two-sided'):
    samp1 = np.asarray(samp1)
    samp2 = np.asarray(samp2)
    pos = count_pos(samp1, samp2)
    neg = count_neg(samp1, samp2)
    M = (pos-neg)/2.
    p = st.binom_test(min(pos,neg), pos+neg, .5, alt)
    return M, p


# Корреляция по Пирсону
def correlate(samp1, samp2):
    if len(samp1) != len(samp2):
        return "error data different range"
    size=len(samp1)
    avg1=samp1.mean()
    avg2=samp2.mean()
    sums1, sums2, sums3 = 0, 0, 0
    for i in range(size):
        sums1 += (samp1[i]-avg1)*(samp2[i]-avg2)
    for i in range(size):
        sums2 += (samp1[i]-avg1)**2
    for i in range(size):
        sums3 += (samp2[i]-avg1)**2
    return sums1/math.sqrt(sums2*sums3)


def sort_un(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1, i, -1):
            if arr[j][1] < arr[j-1][1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]

    return arr

# Корреляция по Спирмену
def correlate2(samp1, samp2):
    united=[] # Список для объединённой выборки
    for i in range(len(samp1)):
        united.append(   ("x", samp1[i], i)   )
    for i in range(len(samp2)):
        united.append(   ("y", samp2[i], i)   )
    rang_x = [i for i in range(len(samp1))]
    rang_y = [i for i in range(len(samp2))]
    united = sort_un(united)

    for i in range(len(united)):
        if united[i][0] == "x":
            rang_x[united[i][2]] = i + 1
        if united[i][0] == "y":
            rang_y[united[i][2]] = i + 1

    rang_x = np.array(rang_x)
    rang_y = np.array(rang_y)
    return correlate(rang_x, rang_y)
    

name1, name2 = r"Data/S_004_001.dat", r"Data/S_004_002.dat"

file = open(name1, "r")
samp1 = np.genfromtxt(name1, dtype ='float')
file.close()

file = open(name2, "r")
samp2 = np.genfromtxt(name2, dtype ='float')
file.close()

print(sign_test_2samp(samp1, samp2))


name3, name4 = r"Data/S_004_003.dat", r"Data/S_004_004.dat"
file = open(name3, "r")
samp3 = np.genfromtxt(name3, dtype='float')
file.close()
file = open(name4, "r")
samp4 = np.genfromtxt(name4, dtype='float')
file.close()
print(sign_test_2samp(samp1, samp3))

print(sign_test_2samp(samp1, samp3, 'less'))

print(sign_test_2samp(samp1, samp3, 'greater'))

print('Mann  ', st.mannwhitneyu(samp1, samp2))

alfa = []
beta = []
for i in range(999):
    alfa.append(float(np.random.rand()))
    beta.append(float(np.random.rand()))
alfa.append(-60.0)
beta.append(+180.0)
alfa = np.array(alfa)
beta = np.array(beta)

print('Пиросон: ', correlate(alfa, beta))

print('Спирмен: ', correlate2(alfa, beta))



print(stats.pearsonr(alfa, beta))
print(stats.spearmanr(alfa, beta))
