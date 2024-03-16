# Compare simulated equilibrium distribution with the theoretical one

import numpy as np
from OneA import getChain

# Transition matrix
P = np.array([[0., 1./3., 2./3.], [0.5, 0.5, 0.], [0.25, 0.5, 0.25]])

# Number of steps
n = 10000

x_sim = getChain(n, P)

# Actual equilibrium distribution
pi = np.array([1./3., 1./3., 1./3.])

# Simulated equilibrium distribution
pi_sim = np.array([np.sum(x_sim == i) for i in range(3)]) / n

# Absolute difference error
error = np.abs(pi - pi_sim)

print('Actual equilibrium distribution:', pi)
print('Simulated equilibrium distribution:', pi_sim)
print('Absolute difference error:', error)
