import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

g1, g2 = 2, 2
std_o1, std_o2 = 0.2, 0.2
dt_obs = 0.25
T = 100
dt = 0.01  # Same as Part b) for generating the true signal
N = int(T / dt)
N_obs = int(T / dt_obs)  # Number of observation points

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

# Generate observations based on the observational equation
v1 = g1*u1[::int(dt_obs/dt)] + np.random.normal(0, std_o1, N_obs)
v2 = g2*u2[::int(dt_obs/dt)] + np.random.normal(0, std_o2, N_obs)

# Initialize Kalman filter parameters
# u1_est, u2_est = [0.5], [0.5]  # Starting with the initial true condition for simplicity
u1_prior, u2_prior = [0.5], [0.5]  # For plotting
# P1, P2 = [1], [1]  # Initial error covariance estimates, arbitrarily chosen
P1_prior, P2_prior = [0.1], [0.1]  # For plotting


u1_est, u2_est = np.zeros(N_obs), np.zeros(N_obs)
P1, P2 = np.ones(N_obs)*0.1, np.ones(N_obs)*0.1  # Initial error covariance estimates


# Observation model and process covariance matrices
H = np.array([[g1, 0], [0, g2]])
R = np.diag([std_o1**2, std_o2**2])
Q = np.diag([sigma1**2*dt_obs, sigma2**2*dt_obs])  # Process noise covariance

# Change the inner model
F11, F22 = -5, -5
F12, F21 = 0.9, -0.8
# Kalman filter update loop
for i in range(1, N_obs):
    # Prediction step
    u1_pred = u1_est[i-1] + dt_obs * (F11*u1_est[i-1] + F12*u2_est[i-1])
    u2_pred = u2_est[i-1] + dt_obs * (F21*u1_est[i-1] + F22*u2_est[i-1])

    P1_pred = P1[i-1] + Q[0, 0]
    P2_pred = P2[i-1] + Q[1, 1]

    # For plotting
    u1_prior.append(u1_pred)
    u2_prior.append(u2_pred)
    P1_prior.append(P1_pred)
    P2_prior.append(P2_pred)

    
    # # Update step for u1
    # K1 = P1_pred / (P1_pred + R[0, 0])
    # u1_update = u1_pred + K1 * (v1[i] - g1*u1_pred)
    # P1_update = (1 - K1 * g1) * P1_pred
    
    # # Update step for u2
    # K2 = P2_pred / (P2_pred + R[1, 1])
    # u2_update = u2_pred + K2 * (v2[i] - g2*u2_pred)
    # P2_update = (1 - K2 * g2) * P2_pred
    # Update step
    # Compute Kalman gain
    K1 = P1_pred * H[0,0] / (H[0,0]**2 * P1_pred + R[0,0])
    K2 = P2_pred * H[1,1] / (H[1,1]**2 * P2_pred + R[1,1])
    
    # Update estimate with measurement
    u1_est[i] = u1_pred + K1 * (v1[i] - H[0,0] * u1_pred)
    u2_est[i] = u2_pred + K2 * (v2[i] - H[1,1] * u2_pred)
    
    # Update error covariance
    P1[i] = (1 - K1 * H[0,0]) * P1_pred
    P2[i] = (1 - K2 * H[1,1]) * P2_pred
    
    


# u1_est = np.array(u1_est)
# u2_est = np.array(u2_est)
# P1 = np.array(P1)
# P2 = np.array(P2)


# Observation indices for true signal to match observation times
obs_indices = np.arange(0, len(u1), int(dt_obs/dt))

# Compute RMSE for posterior estimates
rmse_u1 = np.sqrt(np.mean((u1[obs_indices] - u1_est)**2))
rmse_u2 = np.sqrt(np.mean((u2[obs_indices] - u2_est)**2))

# Compute Pattern Correlation
corr_u1, _ = pearsonr(u1[obs_indices], u1_est)
corr_u2, _ = pearsonr(u2[obs_indices], u2_est)

print(f"RMSE for u1: {rmse_u1}, RMSE for u2: {rmse_u2}")
print(f"Pattern Correlation for u1: {corr_u1}, Pattern Correlation for u2: {corr_u2}")

plt.figure(figsize=(14, 10))

# u1 true, prior, posterior
plt.subplot(2, 1, 1)
plt.plot(time, u1, 'k', label='True $u_1$')
plt.scatter(np.arange(len(v1))*dt_obs, v1, color='gray', alpha=0.5, label='Observations $v_1$')
plt.plot(np.arange(len(u1_prior))*dt_obs, u1_prior, 'c--', label='Prior $u_1$')
plt.plot(np.arange(len(u1_est))*dt_obs, u1_est, 'b', label='Posterior $u_1$')
plt.fill_between(np.arange(len(P1))*dt_obs, np.array(u1_est) - np.sqrt(P1), np.array(u1_est) + np.sqrt(P1), color='blue', alpha=0.2, label='Posterior Variance $u_1$')
plt.fill_between(np.arange(len(P1_prior))*dt_obs, np.array(u1_prior) - np.sqrt(P1_prior), np.array(u1_prior) + np.sqrt(P1_prior), color='cyan', alpha=0.2, label='Prior Variance $u_1$')
plt.xlabel('Time')
plt.ylabel('State $u_1$ and Observations $v_1$')
plt.legend()

# u2 true, prior, posterior
plt.subplot(2, 1, 2)
plt.plot(time, u2, 'k', label='True $u_2$')
plt.scatter(np.arange(len(v2))*dt_obs, v2, color='gray', alpha=0.5, label='Observations $v_2$')
plt.plot(np.arange(len(u2_prior))*dt_obs, u2_prior, 'm--', label='Prior $u_2$')
plt.plot(np.arange(len(u2_est))*dt_obs, u2_est, 'r', label='Posterior $u_2$')
plt.fill_between(np.arange(len(P2))*dt_obs, np.array(u2_est) - np.sqrt(P2), np.array(u2_est) + np.sqrt(P2), color='red', alpha=0.2, label='Posterior Variance $u_2$')
plt.fill_between(np.arange(len(P2_prior))*dt_obs, np.array(u2_prior) - np.sqrt(P2_prior), np.array(u2_prior) + np.sqrt(P2_prior), color='magenta', alpha=0.2, label='Prior Variance $u_2$')
plt.xlabel('Time')
plt.ylabel('State $u_2$ and Observations $v_2$')
plt.legend()

plt.tight_layout()
plt.savefig('c_mod.png')
plt.show()

