# Computing-the-SVD-with-Photonic-Chips
Related code for the Paper "Computing the SVD with photonic chips"

The code base consists of two main parts: The Java code implementing the operations counter and the python code used for rum time comparisons and creation of all plots

## Java


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

