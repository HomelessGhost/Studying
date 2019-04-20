import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def calc_analit(x,t,N,n,sum_number):
    Pi=np.pi
    exp=np.exp
    sin=np.sin
    cos=np.cos
    
    w = (3*N+2)*sin((1/2)*Pi*t)*x**2/(N+n-60)+2*N*(t-1)/(N+3)
    
    v = 0
    for q in range(1,sum_number):
        v_n = sin(Pi*q*x)*calc_g_n(t,q,N,n)*(1-exp(-(Pi*q)**2*t))/(Pi*q)**2
        v = v+v_n
        
    return w+v
 
    
def calc_g_n(t,q,N,n):
    Pi=np.pi
    exp=np.exp
    sin=np.sin
    cos=np.cos
    
    return ((3*N+2)/N-(3*N+2)/(N+n-60))*cos((1/2)*Pi*t)*(-Pi**2*q**2*cos(Pi*q)+2*Pi*q*sin(Pi*q)+2*cos(Pi*q)-2)/(Pi**2*q**3)


def solveExplicit(D, f, mu, b0, b1, a, b, h, T, tau):
    A = list(np.arange(a, b+h, h))         # Space grid
    B = list(np.arange(0, T+tau, tau))     # Time grid

    n = len(B)-1
    m = len(A)-1


    u = np.zeros( (n+1, m+1) )  # Matrix u (row - space, column - time)


    # Setting initial values
    for i in range(0, m+1):
        u[0, i] = mu( A[i] )



    # Setting boundary values
    for j in range(0, n+1):
        u[j, 0] = b0( B[j] )
        u[j, m] = b1( B[j] )


    # Parameters for G matrix
    p = D*tau/(h*h)
    q = 1 - 2*D*tau/(h*h)
    r = D*tau/(h*h)

    # Calculating u values
    for j in range(0, n):
        for i in range(1, m):
            u[j+1, i] = p*u[j, i-1] + q*u[j, i] + r*u[j, i+1] + tau*f( B[j], A[i] )

    return u


def solveImplicit(D, f, mu, b0, b1, a, b, h, T, tau):
    A = list(np.arange(a, b + h, h))      # Space grid
    B = list(np.arange(0, T + tau, tau))  # Time grid

    n = len(B)-1
    m = len(A)-1

    u = np.zeros((n + 1, m + 1))  # Matrix u (row - space, column - time)

    # Setting initial values
    for i in range(0, m + 1):
        u[0, i] = mu( A[i] )

    # Setting boundary values
    for j in range(0, n + 1):
        u[j, 0] = b0( B[j] )
        u[j, m] = b1( B[j] )

    p = -D*tau/(h*h)
    q = 1+2*D*tau/(h*h)
    r = -D*tau/(h*h)

    b_cur_arr  = np.zeros(m+1)
    b_prev_arr = np.zeros(m+1)

    f_arr = np.zeros(m+1)


    d = np.empty(m+1)
    for j in range(1, n+1):
        b_prev_arr[0] = b0(B[j-1])
        b_prev_arr[m] = b1(B[j-1])


        b_cur_arr[0] = b0( B[j] )
        b_cur_arr[m] = b1(B[j])

        for i in range(1, m):
            f_arr[i] = f( B[j], A[i] )

        for i in range(0, m+1):
            d[i] = u[j-1, i] - b_prev_arr[i] + b_cur_arr[i] + tau*f_arr[i]

        u[j] = ThomasAlgorithm(p, q, r, d, m)

    return u


# This is a specific Thomas Alghorithm made to solve this particular problem. Do not use it anywhere else
def ThomasAlgorithm(p, q, r, d, m):
    P = np.empty( (m+1) )
    Q = np.empty( (m+1) )

    P[0] = 0
    Q[0] = d[0]

    for i in range(1, m):
        factor = q - p * P[i - 1]

        P[i] = r / factor;
        Q[i] = (d[i] - p * Q[i - 1]) / factor;

    u = np.empty(m+1)

    u[m] = d[m];
    u[0] = d[0];
    for i in range(m-1, 0, -1):
        u[i] = -P[i] * u[i + 1] + Q[i]

    return u;



# Returns f function based on given n, N
def f_builder(n, N):
    def f(t, x):
        return 2*N/(N+3) + (3*N+2)*x*x*np.pi/(2*N)*np.cos(np.pi*t/2) - 2*(3*N+2)/(N+n-60)*np.sin(np.pi*t/2)
    return f

# Returns mu function based on given n, N
def mu_builder(n, N):
    def mu(x):
        return -2*N/(N+3)
    return mu

# Returns b0 function based on given n, N
def b0_builder(n, N):
    def b0(t):
        return 2*N/(N+3)*(t-1)
    return b0

# Returns b1 function based on given n, N
def b1_builder(n, N):
    def b1(t):
        return 2*N/(N+3)*(t-1) + (3*N+2)/(N+n-60)*np.sin(np.pi*t/2)
    return b1


# HW given
n = 62
N = 15
D = 1
a = 0
b = 1
T = 1
h = 0.05
tau = 0.0005

f = f_builder(n, N)
mu = mu_builder(n, N)
b0 = b0_builder(n, N)
b1 = b1_builder(n, N)


u = solveImplicit(D, f, mu, b0, b1, a, b, h, T, tau)

A = list(np.arange(a, b+h, h))     # Space grid
B = list(np.arange(0, T, tau))     # Time grid


u_analit = []
for i in range(0, len(A)):
    u_analit.append( calc_analit( A[i], 0.5, N, n, 100) )



plt.style.use(['seaborn-darkgrid', 'dark_background'])

line1, = plt.plot(A, u_analit, 'g', label='Аналитическое решение')
line2, = plt.plot(A, u[ int(len(B)/2) ], 'ro', markersize=3, label='Явное решение')

plt.legend(handles=[line2, line1])


plt.show()
