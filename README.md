# Computing-the-SVD-with-Photonic-Chips
Related code and data for the Paper "Computing the SVD with photonic chips" by Johannes Maly, Korbinian Neuner and Samarth Vadia

The code base consists of two main parts: The Java code implementing the operations counter and the python code used for runtime comparisons and creation of all plots.

## Java
This contains the code for the operations counter, some utilities and Java implementations of all Algorithm environments in the paper. The operation count polynomials are obtained by:

1. logging the operation counts for different matrix sizes and iteraction counts
2. finding a polynomial that produces the same output

We can then be sure to find the right polynomial by noticing that the real polynomial can't have to high of a degree.

### main/GRKSVD.java
Implementation of the GRK-SVD algorithm . As it is written with the aim of faithfully recreating the pseudocode from the paper it doesn't contain any deflation or convergence checks.
This makes operation counting possible as the number of operations now only depends on size and the number of performed qr-iterations.

### main/Logger.java
Tracks `Operation` logs in named sections with global timestep support.  
Supports checkpointing, enabling/disabling logging, and generating simplified parallelized logs.  
Provides summary printing of operation counts and timestep ranges for each section.

### main/Main.java
Main class for testing and analyzing SVD algorithms on random matrices. You need to uncomment the specific algorithm you want to test.
Supports performance benchmarking, operation counting, and generating polynomial representations of logged operations outputted in LaTeX format.  
Includes hybrid and digital QR/GRK SVD implementations with optional parallelized logging.
Also generates the `timecheckGRK` and `timecheckQR` datasets.

### main/PhotonicGRKSVD.java
Implements the GRK-SVD optimized for photonic chips (GRK-SVD H).
Simulates the photonic chip with `OneDiagonalChipSimulator` to compute with unitary matrices.  

### main/PhotonicQRSVD.java
Implements the QR-SVD optimized for photonic chips (QR-SVD H).
Simulates the photonic chip with `OneDiagonalChipSimulator` to compute with unitary matrices.  

### main/QRSVD.java
Implements QR-SVD using householder reflections.

### utils/MathUtils.java
Provides mathematical operations with integrated `Logger` tracking for additions, multiplications, divisions, and square roots.  
Supports optional parallelized (GPU-style) operations and conversion of 3D integer tensors into polynomial coefficient form.  
Includes rational matrix inversion and matrix-vector multiplication utilities.

### utils/Matrix.java
Represents a 2D double-precision matrix with support for creation, random generation, identity, and Givens rotations.  
Includes operations like addition, multiplication, scalar multiplication, transposition, embedding, submatrices, Frobenius norm, and copy.  
Integrates with `Logger` and `MathUtils` for operation counting, supports parallelized computation, and provides specialized methods for embedded QR/SVD transformations.

### utils/OneDiagonalChipSimulator.java
Simulates a photonic chip that applies a sequence of 2×2 rotation matrices arranged along one diagonal.  

- **Preset & activate rotations**: Configure Givens rotations into the chip, then "activate" to apply them.
- **Operation counting**: Integrates with `Logger` to track chip configuration (`CHIP_CONF`) and multiplications (`CHIP_MULT`).

### utils/Operation.java
Represents a single logged operation in the system, primarily for tracking arithmetic or chip operations during matrix computations.
This class is used in conjunction with Logger to track all arithmetic and matrix operations for profiling or simulation purposes.

### utils/Rational.java
Represents an arbitrary precision fraction together with some helpful functionality. Used by `MathUtils` to compute the operation count polynomials.


## Python

### grksvd.py
This script implements the GRK-SVD algorithm explicitly including the convergence checks omitted in the Java version.
Utilizes block deflation.
Supports optional U/V computation, singular value sorting, and convergence tracking.

### svdviaqr.py
Implements the QR-SVD algorithm. Includes a custom QR factorization with Householder reflections and utilities to measure matrix diagonality.

### data.py
Creates the datasets 'GRKData.text', 'AltQRData.txt' and 'svd_timings750.txt'.

### plots.py
Draws all plots from the paper. Includes functions returning runtime and energy estimates using the operation counts.
Relative time and energy costs of operation can easily be adjusted. Also includes convergence tracking of bot GRK-SVD and QR-SVD

## Data
Datasets requiring a lot of time to generate. Format is as follows:
1. Every line starts with the size of (square) matrices being tested
2. a space
3. data is then numbers interspaced with spaces
4. End of line. Next line has next size of test matrices

`timecheckGRK.txt` Runtimes for the Java implementation of GRK-SVD D-SC (in ms)
`timecheckQR.txt` Runtimes for the Java implementation of QR-SVD D-SC (in ms)
These two are used in Figure 8

`GRKData.txt` Number of iterations needed by GRK-SVD in the qr-iteration step to converge to within 1e-6
`AltQRData.txt` Number of iterations needed by QR-SVD to converge to within 1e-6
These two are used in Figure 4

`svd_timings750` Runtime of the LAPACK routines gesdd and gesvd for sizes from 5 to 750 (in ms). First half of line is gesdd, second half gesvd
This is used in Figure 9


