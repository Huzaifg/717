import numpy as np

# integrand


def f(x, y):
    return np.exp(x)*y**x


# Integration limits
x_min = 0
x_max = 1

y_min = 0
y_max = 2

area = (x_max - x_min) * (y_max - y_min)
# Run loop in increaments of 100 numbers till 95% confidence interval is within 0.05
N = 1000
while True:
    x = np.random.uniform(x_min, x_max, N)
    y = np.random.uniform(y_min, y_max, N)

    # Evaluate f(x,y) for
    f_eval = f(x, y)

    exp_f = np.mean(f_eval)
    var_f = np.var(f_eval)
    I = area * exp_f
    # Using slide 8 of Lecture 2 to ensure that 1.96 * Sn / sqrt(N) <= 0.05
    conf = 1.96 * np.sqrt(var_f / N)

    print(f"Number of samples: {N}")
    print(f"Sample mean: {exp_f}, Sample variance: {var_f}")
    print(f"Confidence interval: {conf}")
    print(f"Integral: {I}")

    # Using slide 8 of Lecture 2 to ensure that 1.96 * Sn / sqrt(N) <= 0.05
    if conf <= 0.05:
        break
    else:
        N += 1000
