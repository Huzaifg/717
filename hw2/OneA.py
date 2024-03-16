# Markvo chain simulation with given transition matrix

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


def getChain(n: int, P: np.ndarray) -> np.ndarray:
    """
    Simulate a Markov chain with transition matrix P for n steps.
    """
    x = np.zeros(n, dtype=int)
    # spawn uniform random variables of length n
    u1 = np.random.rand(n)
    # Initial distribution
    alpha = np.array([1./3., 1./3., 1./3.])
    # find smallest j such that alpha_0 + ... + alpha_j >= u1[0]
    j = 0
    while alpha[j] < u1[0]:
        j += 1
        if (j == 2):
            break
    x[0] = j
    for t in range(1, n):
        # find smallest j such that P[x[t-1], 0] + ... + P[x[t-1], j] >= u1[t]
        j = 0
        while np.sum(P[x[t-1], :j+1]) < u1[t]:
            j += 1
            if (j == 2):
                break
        x[t] = j
    return x


if __name__ == '__main__':
    # Transition matrix
    P = np.array([[0., 1./3., 2./3.], [0.5, 0.5, 0.], [0.25, 0.5, 0.25]])

    # Number of steps
    n = 10000

    x = getChain(n, P)

    # Plot x_t
    plt.figure(figsize=(18, 3))
    plt.plot(range(len(x)), x)
    plt.title('Markov chain')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.xlim(0, n)
    plt.yticks([0, 1, 2])
    plt.tight_layout()
    plt.savefig('1a_10000.png', dpi=300, bbox_inches='tight')
    plt.show()
