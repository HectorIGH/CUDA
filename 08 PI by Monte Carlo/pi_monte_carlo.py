import numpy as np

def pi_rand(trials, n):
    x = np.random.uniform(-1, 1, size = (trials, n))
    y = np.random.uniform(-1, 1, size = (trials, n))
    return 4 * sum(sum((x**2 + y**2 < 1).astype(int))) / (n * trials)

calc_PI = pi_rand(1_024, 1_024 * 1_00)
print(f'The calculated value of PI is {calc_PI} with an error of {(calc_PI - np.pi) / np.pi}')
