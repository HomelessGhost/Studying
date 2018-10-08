import numpy as np
import math
import scipy.special as sc
import scipy.stats as st
import scipy.integrate as integ
import matplotlib.pyplot as plt
from scipy.special import binom

def sign_test(samp, mu0=0):
    samp = np.asarray(samp)
    pos = np.sum(samp > mu0)
    neg = np.sum(samp < mu0)
    M = (pos-neg)/2.
    p = st.binom_test(min(pos,neg), pos+neg, .5)
    return M, p


def count_pos(samp1, samp2):
    sum = 0
    for i in range(0, len(samp1)):
        if samp1[i] > samp2[i]:
            sum += 1
    return sum


def count_neg(samp1, samp2):
    sum = 0
    for i in range(0, len(samp1)):
        if samp1[i] < samp2[i]:
            sum +=1
    return sum


def sign_test_2samp(samp1, samp2):
    samp1 = np.asarray(samp1)
    samp2 = np.asarray(samp2)
    pos = count_pos(samp1, samp2)
    neg = count_neg(samp1, samp2)
    M = (pos-neg)/2.
    p = st.binom_test(min(pos, neg), pos+neg, .5)
    return M, p




size=100
y1=np.random.rand(size)
y2=[]
for i in range(0, size):
    y2.append(np.random.rand(1)*0.95)
y2 = np.array(y2)

print(f"y1 mean: {y1.mean()}")
print(f"y2 mean: {y2.mean()}")

print(sign_test_2samp(y1, y2))

val=0
for i in range(0, 9):
    val+=binom(10, i)
val=val*math.pow(2, -10)
print(val)

val=0
for i in range(8, 11):
    val+=binom(10, i)
val=val*math.pow(2, -10)
print(val)

print(st.binom_test(2,10,.5))
