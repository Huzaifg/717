# Time evolution of mean and variance of complex OU process

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# IGNORE THIS BLOCK - JUST SETS PLOTTING PARAMETERS
# ==============================================================================
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.family': 'sans-serif',
})
# ==============================================================================


# Parameters
a = 1
omega = 0.5
f = 0.1 + 0.1j
sigma = 0.2
x_0 = 1 + 1j
x_0_prime_squared = 2 + 2j

# Time vector
t = np.arange(0, 5, 0.01)

# Simulate the mean <x_t>
x_t = x_0 * np.exp((-a + 1j*omega)*t) + (f / (a - 1j*omega)
                                         ) * (1 - np.exp((-a + 1j*omega)*t))

# Simulate the variance of the derivative <x_t'^2>
x_t_prime_squared = x_0_prime_squared * np.exp(-2*(a + 1j*omega)*t) + (
    sigma**2 / (2*(a + 1j*omega))) * (1 - np.exp(-2*(a + 1j*omega)*t))

# Plot the real parts of the mean and variance
plt.figure(figsize=(14, 7))

# Plotting the mean
plt.subplot(1, 2, 1)
plt.plot(t, x_t.real, label='Real part of $\\langle x_t \\rangle$')
plt.title('Mean of complex OU process')
plt.xlabel('Time')
plt.ylabel('Mean')
plt.legend()

# Plotting the variance
plt.subplot(1, 2, 2)
plt.plot(t, x_t_prime_squared.real,
         label='Real part of $\\langle x_t\'^2 \\rangle$')
plt.title('Variance of complex OU process')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.legend()

plt.tight_layout()
plt.savefig('3.png', dpi=300)
plt.show()
