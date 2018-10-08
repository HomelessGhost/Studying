import numpy as np
from scipy.special import binom
import scipy.stats as st


def count_pos_neg(sample1, sample2):
    if len(sample1) != len(sample2):
        raise Exception('Samples have different lengths')
    pos, neg = 0, 0
    for i in range(0, len(sample1)):
        if sample1[i] - sample2[i] > 0:
            pos += 1
        else:
            neg += 1
    return pos, neg


def binom_tail_sum(stat_value, sample_length, **kwargs):
    side = kwargs.get('side', 'left')
    sum = 0
    if side == 'left':
        for i in range(0, stat_value + 1):
            sum += binom(sample_length, i)
    else:
        for i in range(stat_value, sample_length + 1):
            sum += binom(sample_length, i)

    sum *= 2 ** (-sample_length)
    return sum


sample1 = np.genfromtxt('../Data/S2_04_001.dat', dtype ='float')
sample2 = np.genfromtxt('../Data/S2_04_002.dat', dtype ='float')


pos, neg = count_pos_neg(sample2, sample1)
print(pos, neg)
sum = binom_tail_sum(neg, len(sample1), side='left')
print(sum)




