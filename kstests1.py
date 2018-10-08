import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest


mass_num = 100 # Количество элементов в выборке

for i in range(0, 101): # Количество сгенерированных выборок
    data = np.random.normal(0, 1, 100)
    test = kstest(data, 'norm')
    if test[1] < 0.95:
        print(test[1])
        data.sort()
        len_data = len(data)
        for i in range(0, len_data-1):
            x = [data[i], data[i+1]]
            y = [i/len_data, i/len_data]
            plt.plot(x, y, color = 'b')


# Код работает странно. Со временем работа значительно замедляется. Причина пока что остаётся неясной.
# Можно попробовать поискать причину
plt.title('Труба')
plt.show()
