#!/usr/bin/env python
# coding: utf-8

# In[12]:


import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#Бучинчик Павел          | 1 | 500 | 0 | 1 |  


# In[13]:


sample_len = 500
iters = 1000
ipdf = "sqrt(uniform)"

def square_under_bin(a, b):
    return b*b-a*a

def bin_number(n):
    return np.sqrt(n)/3

def D_rule(w, p):
    return np.max([np.fabs(w[i]-p[i]) for i in range(len(w))])


# In[14]:


D_list = []

for i in range(iters):
    un = st.uniform.rvs(loc=0, scale=1, size=sample_len)
    un_sqrt = [np.sqrt(un[i]) for i in range(len(un))]

    n, bins, _ = plt.hist(un_sqrt, int(bin_number(sample_len)))
    n_normed = [n[i]/sample_len for i in range(len(n))]

    D_prime = D_rule(n_normed, [square_under_bin(bins[i], bins[i+1]) for i in range(len(bins)-1)])
    #print(D_prime)
    D_list.append(D_prime)
    


# In[15]:


D_list = np.sort(D_list)

Y = [i/len(D_list) for i in range(len(D_list))] # F(x)

plt.plot(D_list, Y, ".")
plt.show()


# In[17]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

q1 = find_nearest(Y, 0.9)
q2 = find_nearest(Y, 0.95)

print(D_list[q1], D_list[q2])


# In[30]:


un = st.uniform.rvs(loc=0, scale=1, size=sample_len)
un_sqrt = [np.sqrt(un[i]) for i in range(len(un))]

n, bins, _ = plt.hist(un_sqrt, int(bin_number(sample_len)))
n_normed = [n[i]/sample_len for i in range(len(n))]

D_prime = D_rule(n_normed, [square_under_bin(bins[i], bins[i+1]) for i in range(len(bins)-1)])
print(D_prime)

X_idx = find_nearest(D_list, D_prime)
print(1-Y[X_idx])


# In[31]:


norm = st.norm.rvs(loc=0, scale=1, size=sample_len)

n, bins, _ = plt.hist(norm, int(bin_number(sample_len)))
n_normed = [n[i]/sample_len for i in range(len(n))]

D_prime = D_rule(n_normed, [square_under_bin(bins[i], bins[i+1]) for i in range(len(bins)-1)])
print(D_prime)

X_idx = find_nearest(D_list, D_prime)
print(1-Y[X_idx])


# In[32]:


norm = st.cauchy.rvs(loc=0, scale=1, size=sample_len)

n, bins, _ = plt.hist(norm, int(bin_number(sample_len)))
n_normed = [n[i]/sample_len for i in range(len(n))]

D_prime = D_rule(n_normed, [square_under_bin(bins[i], bins[i+1]) for i in range(len(bins)-1)])
print(D_prime)

X_idx = find_nearest(D_list, D_prime)
print(1-Y[X_idx])


# In[44]:


un = st.uniform.rvs(loc=0, scale=1.15, size=sample_len)
un_sqrt = [np.sqrt(un[i]) for i in range(len(un))]

n, bins, _ = plt.hist(un_sqrt, int(bin_number(sample_len)))
n_normed = [n[i]/sample_len for i in range(len(n))]

D_prime = D_rule(n_normed, [square_under_bin(bins[i], bins[i+1]) for i in range(len(bins)-1)])
print(D_prime)

X_idx = find_nearest(D_list, D_prime)
print(1-Y[X_idx])

