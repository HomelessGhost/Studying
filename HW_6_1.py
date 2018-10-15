import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def do_bootstrap(sample, accuracy, stat_func):
    size = len(sample)
    stat_D = []
    for i in range(accuracy):
        rand_samp = []
        for i in range(size):
            rand_samp.append(np.random.choice(sample))
        rand_samp = np.array(rand_samp)
        stat_D.append(stat_func(rand_samp))
    return stat_D


def se_boot(bootstrap_D):
    bootstrap_D = np.array(bootstrap_D)
    N = len(bootstrap_D)
    avg = bootstrap_D.mean()
    return np.sum([(bootstrap_D[i] - avg)**2 for i in range(N)])/N


def find_nearest_inv(array, value):
    Y = [i/len(array) for i in range(len(array))]
    return find_nearest(Y, value)


def find_nearest(array, value):
    array = np.asarray(array)
    array = np.sort(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_quantile(statistics, p_val):
    sorted_statistics = np.sort(statistics)
    Y = [i / len(sorted_statistics) for i in range(len(sorted_statistics))]
    return sorted_statistics[find_nearest(Y, p_val)]


def my_cdf(x_data, color):
    x_data = np.sort(x_data)
    len_data = len(x_data)
    for i in range(0, len_data - 1):
        x = [x_data[i], x_data[i + 1]]
        y = [i / len_data, i / len_data]
        plt.plot(x, y, color=color)


def hw_6_1():
    D = []
    for i in range(10000):
        smp = np.random.normal(5, 1, 1000)
        stat = exp_statistics(smp)
        D.append(stat)

    sample = np.random.normal(5, 1, 1000)
    bootstrap_D = do_bootstrap(sample, 10000, exp_statistics)

    # plt.hist(D, bin_number(10000), color='red', alpha=0.5)
    # plt.hist(bootstrap_D, bin_number(10000), color='blue', alpha=0.5)
    cdf_print(D, 'b.')
    cdf_print(bootstrap_D, 'r.')

    print('Квантиль 0.025: ', find_quantile(bootstrap_D, 0.025))
    print('Квантиль 0.975: ', find_quantile(bootstrap_D, 0.975))
    print(se_boot(bootstrap_D))
    plt.show()


def hw_6_2():
    D = []
    for i in range(10000):
        smp = np.random.uniform(0, 1, 1000)
        stat = np.max(smp)
        D.append(stat)

    sample = np.random.uniform(0, 1, 1000)
    bootstrap_D = do_bootstrap(sample, 1000, lambda array: np.max(array))

    sns.distplot(D)
    # plt.hist(D, bin_number(10000), color='red', alpha=0.5)
    # plt.hist(bootstrap_D, bin_number(10000), color='blue', alpha=0.5)
    # my_cdf(D, 'blue')
    # my_cdf(bootstrap_D, 'red')

    print('Квантиль 0.025: ', find_quantile(bootstrap_D, 0.025))
    print('Квантиль 0.975: ', find_quantile(bootstrap_D, 0.975))
    print(se_boot(bootstrap_D))
    plt.show()

hw_6_2()