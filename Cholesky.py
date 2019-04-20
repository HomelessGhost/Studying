import numpy as np

def generateWhiteNoize(N):
    return np.random.randn(N)

def corFunc(t1, t2):
    return np.cos(t1-t2)


grid = np.arange(0, 2*np.pi, 0.01)

C = np.empty( (len(grid), len(grid)) )

for i in range(0, len(grid)):
    for j in range(0, len(grid)):
        C[i, j] = corFunc( grid[i], grid[j] )

L = np.linalg.