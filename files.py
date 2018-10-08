import numpy as np
import matplotlib.pyplot as plt
import math

data = np.genfromtxt('../Data/S2_04_001.dat', dtype = 'float')
quantity = list(range(len(data)))
data.sort()
len_data = len(data)
for i in range(0, len_data-1):
    x = [data[i], data[i+1]]
    y = [i/len_data, i/len_data]
    plt.plot(x, y, color = 'b')





#data_1 = np.array([data[x][0] for x in range(len(data))])
#data_2 = np.array([data[x][1] for x in range(len(data))])


#bars = math.floor((math.sqrt(len(data))))
#plt.scatter(quantity, data, color = 'r', s = 5)

#plt.scatter(quantity, data_1, color = 'r', s = 5)
#plt.scatter(quantity, data_2, color = 'b', s = 5)

#plt.hist(data, bars)


plt.title('Что это за выборка?')
#plt.yscale('log')
plt.show()
