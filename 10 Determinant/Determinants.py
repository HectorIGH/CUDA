#!/usr/bin/env python
# coding: utf-8

import numpy as np

def permutations(original_list):
    size = len(original_list)
    permutations_list = []
    if (size == 0):
        return []
    if (size == 1):
        return original_list
    for i in range(size):
        current_element = original_list[i]
        remaining_list = np.append(original_list[:i], original_list[i + 1:])
        for permut in permutations(remaining_list):
            permutations_list.append(np.append(current_element, permut))
    return np.array(permutations_list)

def bubble_counter(Sn):
    rows, cols = Sn.shape
    counter = np.zeros(rows)
    for index, permutation in enumerate(Sn):
        size = len(permutation)
        for i in range(size):
            for j in range(size):
                if permutation[i] > permutation[j] and i < j:
                    #permutation[i], permutation[j] = permutation[j], permutation[i]
                    counter[index] += 1
    return counter

def determinant_by_permutations(matrix):
    n = matrix.shape[0]
    
    Sn = permutations(np.arange(n))
    
    inversions = bubble_counter(Sn)
    
    sign = np.array([-1] * inversions.shape[0]) ** inversions
    
    holder_for_sum = np.zeros(Sn.shape[0])
    for index, permutation in enumerate(Sn):
        current_product = sign[index]
        for origin, sigma in enumerate(permutation):
            current_product *= matrix[origin][sigma]
        holder_for_sum[index] = current_product
    dt = sum(holder_for_sum)
    return dt, holder_for_sum

def determinant_by_minors(matrix):
    n = matrix.shape[0]
    if n == 1:
        return matrix[0][0]
    sums = 0.0
    for i in range(n):
        minor = np.delete(matrix, 0, 0) # removing row for minor
        minor = np.delete(minor, i, 1) # removing column for minor
        sums += matrix[0][i] * determinant_by_minors(minor) * (-1) ** i
    return sums

def cofactor_matrix(matrix):
    cofactors = np.zeros_like(matrix)
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            minor = np.delete(matrix, i, 0) # removing row for minor
            minor = np.delete(minor, j, 1) # removing column for minor
            cofactors[i][j] = determinant_by_minors(minor) * (-1) ** (i + j)
    return cofactors

def factorial(n):
    if n == 1:
        return n
    return n * factorial(n - 1)


n = 8
matrix = np.array([42,68,35,1,70,25,79,59,63,65,6,46,82,28,62,92,96,43,28,37,92,5,3,54,93,83,22,17,19,96,48,27,72,39,70,13,68,100,36,95,4,12,23,34,74,65,42,12,54,69,48,45,63,58,38,60,24,42,30,79,17,36,91,43])
matrix = matrix[:n*n].reshape((n, n))
print(matrix)

a, b = determinant_by_permutations(matrix)

permutationes = [factorial(n) for n in range(2, 9)]

times_p = [4, 8, 6, 11, 46, 321, 3000]
times_c = [0.071, 0.075, 0.061, 0.071, 0.092, 0.142, 0.280]

import matplotlib.pyplot as plt
from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'qt')
rcParams['figure.figsize'] = 16, 8
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['lines.linewidth'] = 3
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

n = [i for i in range(2, 9)]

py = plt.scatter(n, times_p, s = 200, c = 'red', marker = 'o', label = 'Sequential time')
py = plt.plot(n, times_p, c = 'red')

cu = plt.scatter(n, times_c, s = 300, c = 'blue', marker = '*', label = 'Parallel time')
cu = plt.plot(n, times_c, c = 'blue')
#plt.ylim(top = max(times_p))
#plt.ylim(bottom = 0)
plt.xlabel('Matrix dimension. N x N')
plt.ylabel('Time taken (ms)')
plt.legend()
