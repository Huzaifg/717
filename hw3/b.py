import numpy as np
import matplotlib.pyplot as plt

# System parameters
F11, F22 = -1, -1
F12, F21 = 0.5, -0.5
sigma1, sigma2 = 0.2, 0.5

# Initial conditions
u1_0, u2_0 = 0.5, 0.5

# Simulation parameters
T = 100  # total time
dt = 0.01  # time step
N = int(T / dt)  # number of steps

# Preallocate solution arrays
time = np.arange(0, T, dt)
u1 = np.zeros(N)
u2 = np.zeros(N)
u1[0], u2[0] = u1_0, u2_0

# Generate the time series
for i in range(1, N):
    u1[i] = u1[i-1] + (F11*u1[i-1] + F12*u2[i-1])*dt + sigma1*np.sqrt(dt)*np.random.normal()
    u2[i] = u2[i-1] + (F21*u1[i-1] + F22*u2[i-1])*dt + sigma2*np.sqrt(dt)*np.random.normal()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(time, u1, label='$u_1(t)$')
plt.plot(time, u2, label='$u_2(t)$')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.legend()
plt.savefig('b.png')
plt.show()
