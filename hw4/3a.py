import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 1.0
f = 1.0
sigma = 0.5
dt = 0.01  # Simulation timestep
observation_dt = 0.25  # Observation interval
total_time = 100
observation_variance = 0.01
N = int(total_time / dt)
num_observations = int(total_time / observation_dt)

# Initial conditions
u = 0.0

# Pre-allocate
times = np.arange(0, total_time, dt)
observations_times = np.arange(0, total_time, observation_dt)
u_values = np.zeros(N)
observations = np.zeros(num_observations)

np.random.seed(0)

# Simulation using Euler-Maruyama method
for i in range(N):
    u += (-a * u + f) * dt + sigma * np.sqrt(dt) * np.random.randn()
    u_values[i] = u
    if i % int(observation_dt / dt) == 0:
        obs_index = i // int(observation_dt / dt)
        observations[obs_index] = u + np.sqrt(observation_variance) * np.random.randn()

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(times, u_values, label='True $u(t)$')
plt.scatter(observations_times, observations, color='r', label='Observations')
plt.xlabel('Time')
plt.ylabel('$u(t)$')
plt.title('Time series and observations')
plt.legend()
plt.savefig('3a.png',dpi=600)
plt.show()
