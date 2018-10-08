import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


# Яхновский Алексей | 1 | 250 | 1 | 3 |


# Правило Стёрджеса или Большого пальца
def bin_number(n):
    return int(1 + np.floor(np.log2(n)))
#    return int(np.sqrt(n)/3)


#D[m.n]
def D_rule(w, p):
    return np.sqrt(np.sum(    [(w[i] - p[i]) ** 2 for i in range(len(w))]    ))
#    return np.max([np.fabs(w[i]-p[i]) for i in range(len(w))])
#    return np.sum([   ((w[i] - p[i]) ** 2)/(p[i]*(1 - p[i])) for i in range(len(w))  ])

def square_under_bin(a, b):
    if a > 1 or b < 0:
        return 0
    else:
        return b * b - a * a
#    return b - a


sample_len = 250
iters = 2000


def build_statistic_distribution(distribution):
    D_list = []

    for i in range(iters):
        un = st.uniform.rvs(loc=0, scale=1, size=sample_len)
        un_sqrt = [np.sqrt(un[i]) for i in range(len(un))]

        n, bins, _ = plt.hist(un_sqrt, bin_number(sample_len))
#        n, bins, _ = plt.hist(un, bin_number(sample_len))
        n_normed = [n[i]/sample_len for i in range(len(n))]

        plt.clf()
        D_prime = D_rule(n_normed, [square_under_bin(bins[i], bins[i+1]) for i in range(len(bins)-1)])
        D_list.append(D_prime)

    D_list = np.sort(D_list)
    Y = [i/len(D_list) for i in range(len(D_list))] # F(x)
    with open(distribution, 'w') as stream:
        for i in range(len(D_list)):
            print(D_list[i], Y[i], sep='   ', file=stream)


def show_statisctic_distribution(distribution):
    x, y = load_statistic_distribution(distribution)
    plt.plot(x, y, ".")
    plt.show()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def load_statistic_distribution(distribution):
    data = np.genfromtxt(distribution, dtype='float')
    data_len = len(data)
    x = np.array([data[x][0] for x in range(data_len)])
    y = np.array([data[x][1] for x in range(data_len)])
    return x, y


def find_quantile(propability, distribution):
    x, y = load_statistic_distribution(distribution)
    return x[find_nearest(y, propability)]


def test_distribution(sample, distribution):
    n, bins, _ = plt.hist(sample, bin_number(len(sample)))
    n_normed = [n[i]/sample_len for i in range(len(n))]

    D_prime = D_rule(n_normed, [square_under_bin(bins[i], bins[i+1]) for i in range(len(bins)-1)])
    x, y = load_statistic_distribution(distribution)

    X_idx = find_nearest(x, D_prime)
    return 1-y[X_idx], D_prime


un = st.uniform.rvs(loc=0, scale=0.8, size=sample_len)

#un = st.cauchy.rvs(loc=0, scale=1, size=sample_len)
#un = st.norm.rvs(loc=0, scale=1, size=sample_len)
# un_sqrt = [np.sqrt(un[i]) for i in range(len(un))]
#
# p, D_pr = test_distribution(un_sqrt, r'D_distribution.txt')
# print(p, D_pr)

#build_statistic_distribution(r'D_distribution_Luda100.txt')
show_statisctic_distribution(r'D_distribution.txt')

# print(find_quantile(0.1, r'D_distribution_Luda100.txt'))
# print(find_quantile(0.05, r'D_distribution_Luda100.txt'))
