# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:14:13 2025

@author: korbi
"""
import sys
import numpy as np

def qr(matrixA):
    # Make a copy of the input matrix and ensure it's of type float
    A = matrixA.copy().astype(float)
    m, n = A.shape # Get matrix dimensions
    Q = np.eye(m) # Initialize Q as an identity matrix of size m x m
    
    # Loop over each column up to the smaller dimension of the matrix
    for k in range(min(m, n)):
        a = A[k:m, k:k+1].copy() # Extract the k-th column starting from row k
        sign = np.sign(a[0, 0]) if a[0, 0] != 0 else 1.0 # Determine the sign of the first element to avoid cancellation
        
        # Construct the first basis vector e1
        e1 = np.zeros_like(a)
        e1[0, 0] = 1.0
        
        # Compute the Householder vector v
        v = a + sign * np.linalg.norm(a) * e1
        v = v / np.linalg.norm(v)  # Normalize the Householder vector
        
        # Outer product of v to construct the Householder reflection matrix
        Ua = np.atleast_2d(np.dot(v, v.T))
        
        # Initialize Hk as an identity matrix
        Hk = np.eye(m)
        
        # Apply the reflection to the submatrix (rows and columns from k to m)
        for i in range(k, m):
            for j in range(k, m):
                Hk[i, j] = Hk[i,j] - 2 * Ua[i - k, j - k]
        
        # Update A by applying the Householder transformation
        A = np.dot(Hk, A)
        # Accumulate the orthogonal transformations into Q
        Q = np.dot(Q, Hk)

    R = A # After all transformations, A becomes the upper-triangular R
    return Q, R # Return orthogonal matrix Q and upper-triangular matrix R

# Computes a measure of how non-diagonal the matrix is
def diagonality(matrixA):
    A = np.array(matrixA)
    np.fill_diagonal(A, 0) # Zero out the diagonal
    
    # Return the maximum absolute value of off-diagonal elements
    return np.max(np.abs(A))

def naive_svd(matrixA):
    m, n = matrixA.shape
    
    # Set a small threshold for convergence
    delta = max(matrixA.flat[matrixA.argmax()], 1) * sys.float_info.epsilon
    delta = 10e-6 #Same override as in GRK SVD, for the same reason
    
    U = np.eye(m) # Initialize U as identity
    V = np.eye(n) # Initialize V as identity
    D = np.array(matrixA) # Copy of the input matrix to be diagonalized
    
    errors = [] # Store error (off-diagonal max) at each iteration
    limit = 10000 # Maximum number of iterations
    index = 0 # Iteration counter
    
    # Iteratively diagonalize D using QR decompositions
    while diagonality(D) > delta:
        Q, R = qr(D)         # QR decomposition of D
        U = np.dot(U, Q)     # Accumulate transformations into U
        P, S = qr(R.T)       # QR decomposition of R transpose
        V = np.dot(P.T, V)   # Accumulate transformations into V
        D = S.T              # Update D
        index += 1
        errors.append(diagonality(D)) # Track off-diagonal magnitude for error convergence plot
        if(index > limit):   # Stop if too many iterations
            break
    print(index) # Print total iterations
    return U, D, V, index, errors # Return SVD factors and diagnostics

