import numpy as np
import matplotlib.pyplot as plt

# System parameters
F11, F12, F21, F22 = -0.5, 0.1, 0.2, -0.6  # Ensure F11 + F22 < 0
sigma1, sigma2 = 0.1, 0.2

# Observation parameters
g1, g2 = 2.0, 2.0
sigma_o1, sigma_o2 = 0.2, 0.2

# Generate true signal
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


# Initialize state vector and covariance matrix
x = np.zeros(2)
P = np.eye(2)

# Lists to store results
posterior_mean_u1 = []
posterior_mean_u2 = []
posterior_var_u1 = []
posterior_var_u2 = []
prior_mean_u1 = []
prior_mean_u2 = []
prior_var_u1 = []
prior_var_u2 = []

for n in range(N):
   # Time update (prediction)
   x = np.array([x[0] + (F11 * x[0] + F12 * x[1]) * dt,
                 x[1] + (F21 * x[0] + F22 * x[1]) * dt])
   P = P + dt * np.array([[F11 * P[0, 0] + F12 * P[0, 1], F11 * P[0, 1] + F12 * P[1, 1]],
                          [F21 * P[0, 0] + F22 * P[0, 1], F21 * P[0, 1] + F22 * P[1, 1]]]) \
       + np.array([[sigma1 ** 2 * dt, 0],
                   [0, sigma2 ** 2 * dt]])

   # Store prior mean and variance
   prior_mean_u1.append(x[0])
   prior_mean_u2.append(x[1])
   prior_var_u1.append(P[0, 0])
   prior_var_u2.append(P[1, 1])

   # Measurement update (correction)
   z = np.array([u1[n] + np.random.randn() * sigma_o1,
                 u2[n] + np.random.randn() * sigma_o2])
   H = np.array([[g1, 0],
                 [0, g2]])
   R = np.array([[sigma_o1 ** 2, 0],
                 [0, sigma_o2 ** 2]])
   y = z - np.dot(H, x)
   S = np.dot(H, np.dot(P, H.T)) + R
   K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
   x = x + np.dot(K, y)
   P = P - np.dot(K, np.dot(S, K.T))

   # Store posterior mean and variance
   posterior_mean_u1.append(x[0])
   posterior_mean_u2.append(x[1])
   posterior_var_u1.append(P[0, 0])
   posterior_var_u2.append(P[1, 1])

# Compute RMSEs and pattern correlations
rmse_prior_u1 = np.sqrt(np.mean((np.array(prior_mean_u1) - u1) ** 2))
rmse_prior_u2 = np.sqrt(np.mean((np.array(prior_mean_u2) - u2) ** 2))
rmse_posterior_u1 = np.sqrt(np.mean((np.array(posterior_mean_u1) - u1) ** 2))
rmse_posterior_u2 = np.sqrt(np.mean((np.array(posterior_mean_u2) - u2) ** 2))

corr_prior_u1 = np.corrcoef(prior_mean_u1, u1)[0, 1]
corr_prior_u2 = np.corrcoef(prior_mean_u2, u2)[0, 1]
corr_posterior_u1 = np.corrcoef(posterior_mean_u1, u1)[0, 1]
corr_posterior_u2 = np.corrcoef(posterior_mean_u2, u2)[0, 1]

print(f"RMSE (prior) for u1: {rmse_prior_u1:.4f}")
print(f"RMSE (prior) for u2: {rmse_prior_u2:.4f}")
print(f"RMSE (posterior) for u1: {rmse_posterior_u1:.4f}")
print(f"RMSE (posterior) for u2: {rmse_posterior_u2:.4f}")
print(f"Pattern correlation (prior) for u1: {corr_prior_u1:.4f}")
print(f"Pattern correlation (prior) for u2: {corr_prior_u2:.4f}")
print(f"Pattern correlation (posterior) for u1: {corr_posterior_u1:.4f}")
print(f"Pattern correlation (posterior) for u2: {corr_posterior_u2:.4f}")

# Plotting
time = np.arange(N) * dt
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(time, u1, label='True signal')
axs[0, 0].plot(time, prior_mean_u1, '--', label='Prior mean')
axs[0, 0].plot(time, posterior_mean_u1, label='Posterior mean')
axs[0, 0].set_title('u1')
axs[0, 0].legend()

axs[0, 1].plot(time, np.sqrt(prior_var_u1), '--', label='Prior std')
axs[0, 1].plot(time, np.sqrt(posterior_var_u1), label='Posterior std')
axs[0, 1].set_title('u1 standard deviation')
axs[0, 1].legend()

axs[1, 0].plot(time, u2, label='True signal')
axs[1, 0].plot(time, prior_mean_u2, '--', label='Prior mean')
axs[1, 0].plot(time, posterior_mean_u2, label='Posterior mean')
axs[1, 0].set_title('u2')
axs[1, 0].legend()

axs[1, 1].plot(time, np.sqrt(prior_var_u2), '--', label='Prior std')
axs[1, 1].plot(time, np.sqrt(posterior_var_u2), label='Posterior std')
axs[1, 1].set_title('u2 standard deviation')
axs[1, 1].legend()

plt.tight_layout()
plt.show()