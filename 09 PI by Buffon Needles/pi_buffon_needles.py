import numpy as np

def pi_buffon_needles(trials, throws):
    x = np.random.random(size = (trials, throws))
    theta = np.random.random(size = (trials, throws)) * np.pi / 2
    return (trials * throws) / sum(sum(x <= 0.5 * np.sin(theta)))

calc_PI = pi_buffon_needles(512, 512)
print(f'The calculated value of PI is {calc_PI} with an error of {(calc_PI - np.pi) / np.pi}')
