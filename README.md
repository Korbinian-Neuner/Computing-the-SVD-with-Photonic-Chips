# Computing-the-SVD-with-Photonic-Chips
Related code for the Paper "Computing the SVD with photonic chips"

The code base consists of two main parts: The Java code implementing the operations counter and the python code used for rum time comparisons and creation of all plots






# Singular Value Decomposition (SVD) — From Scratch via Implicit QR

A fully manual implementation of the **Singular Value Decomposition (SVD)** using classical numerical linear algebra techniques — without relying on `numpy.linalg.svd`.

---

##  What It Does

Given a matrix:

\[
A \in \mathbb{R}^{m \times n}
\]

It computes the decomposition:

\[
A = U \Sigma V^T
\]

Where:

- `U` — orthogonal left singular vectors  
- `Σ` — diagonal matrix of singular values  
- `V` — orthogonal right singular vectors  

All computed manually via classical numerical algorithms.

---


By following the Steps fropm the GRK-Paper

## 1. Bidiagonalization (Householder Reflections)

The matrix is reduced to upper bidiagonal form:

\[
A = P B Q
\]

Using alternating:

- Left Householder reflections (eliminate subdiagonal elements)
- Right Householder reflections (eliminate super-superdiagonal elements)

This reduces the problem to diagonalizing a much simpler matrix.

---

## 2. Implicit Shifted QR Iteration

The bidiagonal matrix is diagonalized using:

-  Implicit QR steps  
-  Wilkinson shift (for faster convergence)  
-  Givens rotations (bulge chasing)

This phase drives the matrix toward diagonal form while preserving singular values.

---

## 3️. Deflation & Block Splitting

When off-diagonal elements become sufficiently small:

- The matrix splits into independent sub-blocks
- Each block is processed recursively
- A queue manages active blocks

This improves stability and efficiency.

---

## 4. Reconstruction of U and V

During QR iteration:

- Left rotations are stored
- Right rotations are stored
- Block-diagonal multiplication reconstructs full orthogonal matrices
- Singular values are:
  - Made positive
  - Sorted descending
  - Matched with corresponding singular vectors

Final reconstruction:
U = P · accumulated_left_rotations
V = accumulated_right_rotations · Q

