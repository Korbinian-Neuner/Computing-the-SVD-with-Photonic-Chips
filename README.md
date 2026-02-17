# Computing-the-SVD-with-Photonic-Chips
Related code for the Paper "Computing the SVD with photonic chips"

The code base consists of two main parts: The Java code implementing the operations counter and the python code used for rum time comparisons and creation of all plots






## SVD.py

This script implements the GRK - algorithm explicitly including the convergence checks omitted in the Java version.

Given a matrix \( A \in \mathbb{R}^{m \times n} \), it computes:

\[
A = U \Sigma V^T
\]

using the following steps:

1. **Bidiagonalization** via Householder reflections  
2. **Implicit shifted QR iteration** (Wilkinson shift + Givens rotations)  
3. **Deflation and block splitting** for efficiency  
4. **Explicit reconstruction** of \( U \) and \( V \)

### Features

- Manual implementation of GRK SVD  
- Optional computation of `U` and/or `V`  
- Recursive block processing using a queue
- Singular value sorting and sign correction  
- Tracking Convergence behaviour during iterations 



Final reconstruction:
U = P · accumulated_left_rotations
V = accumulated_right_rotations · Q

