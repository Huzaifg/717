import numpy as np

mean = np.array([0, 0])
covariance = np.array([[1, 0.5], [0.5, 1]])


Ns = [10**3, 10**4, 10**5]

for N in Ns:
    x = np.random.multivariate_normal(mean, covariance, size=N)

    # Extract all x_1 for which x_2 lies in (0.9, 1.1)
    x_1 = x[(x[:, 1] > 0.9) & (x[:, 1] < 1.1), 0]

    # approximate p(x_1 | x_2 = 1)
    sample_mean = np.mean(x_1)
    sample_variance = np.var(x_1)

    print(f"Number of samples: {N}")
    print(f"Sample mean: {sample_mean}, Sample variance: {sample_variance}")

    # Computing 95% confidence interval
    # From slide 19 Lecture 1 - we get z_x = 1.96 for 95% confidence interval

    lhs_confidence = sample_mean - 1.96 * np.sqrt(sample_variance / N)
    rhs_confidence = sample_mean + 1.96 * np.sqrt(sample_variance / N)
    print(f"95% confidence interval: [{lhs_confidence}, {rhs_confidence}]")
