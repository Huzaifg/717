import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_ = 1.0
b = 0
c = 1
dt = 0.01
T = 10000
N = int(T / dt)
x = 0.5  # Initial condition within (b, c)


np.random.seed(0)

X = np.zeros(N)
X[0] = 0.5  # Start roughly in the middle of the interval

# Simulate the SDE with boundary checking
for i in range(1, N):
    x = X[i-1]
    dW = np.sqrt(dt) * np.random.randn()
    dx = -lambda_ * (x - (b + c) / 2) * dt + np.sqrt(lambda_ * (x - b) * (c - x)) * dW
    x_new = x + dx
    
    # Reflecting boundary conditions to ensure x stays within (b, c)
    if x_new < b or x_new > c:
        x_new = x - dx  # Reflect x back within the interval

    X[i] = x_new

# Plotting the corrected simulation results
plt.figure(figsize=(10, 6))
plt.hist(X, bins=100, density=True, alpha=0.75)
plt.plot([b, c], [1/(c-b)]*2, 'r--', label='Uniform density')
plt.title('Histogram of X values')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.savefig('1.png',dpi=600)
plt.show()
