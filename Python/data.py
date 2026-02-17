# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 12:38:50 2025

@author: korbi
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import svd as profi_svd
import grksvd
import svdviaqr

# Parameters
n_values = range(5, 751)
num_matrices = 200

def profi_comparison(maxSize, num_matrices):
    times_gesdd = []
    times_gesvd = []

    # Open text file for writing
    output_path = "svd_timings750.txt"
    with open(output_path, "w") as f:
        for n in n_values:
            t_gesdd = []
            t_gesvd = []

            for _ in range(num_matrices):
                A = np.random.randn(n, n)

                # Time gesdd
                start = time.perf_counter()
                profi_svd(A, lapack_driver="gesdd", full_matrices=True)
                end = time.perf_counter()
                t_gesdd.append((end - start) * 1000.0)

                # Time gesvd
                start = time.perf_counter()
                profi_svd(A, lapack_driver="gesvd", full_matrices=True)
                end = time.perf_counter()
                t_gesvd.append((end - start) * 1000.0)
                
            print(n)
            times_gesdd.append(np.mean(t_gesdd))
            times_gesvd.append(np.mean(t_gesvd))

            # Write row: n followed by all times in ms
            f.write(str(n))
            for t in t_gesdd + t_gesvd:
                f.write(f" {t:.6f}")
            f.write("\n")

    output_path


def SizeDependencyGRK(minSize, maxSize, iterationsPerSize):
    medians = []
    datas2 = []
    
    output_path = "GRKData.txt"
    
    with open(output_path, "w") as f:
        for size in range(minSize, maxSize):
            test_matrices = []
            print(f"creating test matrices with size {size}")
            for i in range(iterationsPerSize):
                matrix = np.random.rand(size, size)
                test_matrices.append(matrix)
            
            
            data = []
            for matrix in test_matrices:
                svds, iterations = svd.svd(matrix, False, False, 100000)
                data.append(iterations)
                
            data.sort()
            n = len(data)
            
            if n % 2 == 1:
                median = data[n // 2]
            else:
                mid1 = data[n // 2 - 1]
                mid2 = data[n // 2]
                median = (mid1 + mid2) / 2
            datas2.append(data)
            medians.append(median)
            
            s = str(size)
            for d in data:
                s = s + ' ' + str(d)
            f.write(s)
            
            print(size)
        
        
    datas = list(map(list, zip(*datas2)))
    for m in datas:
        plt.scatter(range(minSize, maxSize), m, alpha=0.05, color='blue')
    
    # fit line: degree=1
    m, b = np.polyfit(range(minSize, maxSize), medians, 1)  # returns slope (m) and intercept (b)
    print("Slope:", m, "Intercept:", b)

    # create points for the line
    y_fit = m * range(minSize, maxSize) + b
    
    
    plt.plot(range(minSize, maxSize), y_fit, color='black', label='Best fit')
    
    plt.title('Required Iterations to reach error of 10e-6')
    plt.xlabel('Dimension')
    plt.ylabel('Iterations')
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    
    return


def SizeDependencyAlternatingQR(minSize, maxSize, iterationsPerSize):
    medians = []
    datas2 = []
    
    output_path = "AltQRData.txt"
    
    with open(output_path, "w") as f:
        for size in range(minSize, maxSize):
            test_matrices = []
            print(f"creating test matrices with size {size}")
            for i in range(iterationsPerSize):
                matrix = np.random.rand(size, size)
                test_matrices.append(matrix)
            
            
            data = []
            for matrix in test_matrices:
                U, D, V, iterations, errors = svdviaqr.naive_svd(matrix)
                data.append(iterations)
                
            data.sort()
            n = len(data)
            
            if n % 2 == 1:
                median = data[n // 2]
            else:
                mid1 = data[n // 2 - 1]
                mid2 = data[n // 2]
                median = (mid1 + mid2) / 2
            datas2.append(data)
            medians.append(median)
            
            s = str(size)
            for d in data:
                s = s + ' ' + str(d)
            f.write(s)
            
            print(size)
        
        
    datas = list(map(list, zip(*datas2)))
    for m in datas:
        plt.scatter(range(minSize, maxSize), m, alpha=0.05, color='blue')
    
    # fit line: degree=1
    m, b = np.polyfit(range(minSize, maxSize), medians, 1)  # returns slope (m) and intercept (b)
    print("Slope:", m, "Intercept:", b)

    # create points for the line
    y_fit = m * range(minSize, maxSize) + b
    
    
    plt.plot(range(minSize, maxSize), y_fit, color='black', label='Best fit')
    
    plt.title('Required Iterations to reach error of 10e-6')
    plt.xlabel('Dimension')
    plt.ylabel('Iterations')
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    
    return



