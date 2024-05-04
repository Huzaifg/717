from scipy.linalg import block_diag
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

# Parameters for Kalman Filter
f_initial_mean = 0.0
f_initial_variance = 1.0

# Initial state
x = np.array([0.0, f_initial_mean])  # Initial [u, f]
P = block_diag(0.01, f_initial_variance)  # Initial covariance matrix

# State transition matrix
A = np.array([[1 - a * dt, dt],
              [0, 1]])

# Process noise covariance
Q = block_diag(sigma**2 * dt, 0)  # No process noise on f

# Observation matrix
H = np.array([[1, 0]])

# Observation noise covariance
R = np.array([[observation_variance]])

# Pre-allocate for estimation tracking
f_estimates = np.zeros(num_observations)

# Simulation using Euler-Maruyama method
for i in range(N):
    u += (-a * u + f) * dt + sigma * np.sqrt(dt) * np.random.randn()
    u_values[i] = u
    if i % int(observation_dt / dt) == 0:
        obs_index = i // int(observation_dt / dt)
        observations[obs_index] = u + np.sqrt(observation_variance) * np.random.randn()




# Kalman Filter Implementation
for i in range(num_observations):
    # Time Update (Predict)
    x = A @ x  # State prediction
    P = A @ P @ A.T + Q  # Covariance prediction

    # Measurement Update (Correct)
    z = observations[i]  # Observation at time t
    y = z - H @ x  # Innovation
    S = H @ P @ H.T + R  # Innovation covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    x = x + K @ y  # State update
    P = (np.eye(2) - K @ H) @ P  # Covariance update

    # Save the estimate of f
    f_estimates[i] = x[1]

# Plotting the estimated value of f over time
plt.figure(figsize=(10, 5))
plt.plot(observations_times, f_estimates, label='Estimated $f$')
plt.axhline(y=f, color='r', linestyle='--', label='True $f$')
plt.title('Estimation of $f$ over time')
plt.xlabel('Time')
plt.ylabel('Estimated $f$')
plt.legend()
plt.savefig('3b.png',dpi=600)
plt.show()
