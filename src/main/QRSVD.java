package main;

import static utils.MathUtils.add;
import static utils.MathUtils.div;
import static utils.MathUtils.mult;
import static utils.MathUtils.sqrt;

import utils.Matrix;
import utils.OneDiagonalChipSimulator;

public class QRSVD {
	
	// High-level iterative SVD using repeated QR decompositions
    // A is decomposed into U * Σ * V^T after 'iterations' sweeps
	public static Matrix[] calcSVD(Matrix A, int iterations) {
		int m = A.getVSize();
		int n = A.getHSize();
		
		// Initialize U and V as identity matrices for accumulation
		Matrix U = Matrix.createIdentity(m);
		Matrix V = Matrix.createIdentity(n);
		
		// Iteratively apply QR decompositions to diagonalize A
		for(int it = 0; it < iterations; it++) {
			// Step 1: QR decomposition of A
			Matrix[] qr = QR_Decomposition(A);
			U = Matrix.multiply(U, qr[0]); // Accumulate left orthogonal transforms
			
			// Step 2: QR decomposition of A^T to handle right side
			qr = QR_Decomposition(qr[1].transpose());
			V = Matrix.multiply(V, qr[0]); // Accumulate right orthogonal transforms
			A = qr[1].transpose();		  	// Update A for next iteration
		}
		
		// Return final SVD components: U, bidiagonal/diagonal A (≈Σ), V^T
		Matrix[] ret = {U, A, V.transpose()};
		return ret;
	}
	
	// Performs QR decomposition of a matrix using Householder reflections
    // Returns {Q, R} where Q is orthogonal and R is upper-triangular
	public static Matrix[] QR_Decomposition(Matrix A) {
		int m = A.getVSize();
		int n = A.getHSize();
		
		// Initialize Q accumulator as identity
		Matrix P = Matrix.createIdentity(m);
		
		// Only iterate over first min(m-1, n) columns
		n = m < n ? m - 1: n;
		
		for(int k = 0; k < n; k++) {
			// Extract the k-th column vector starting from diagonal
			Matrix a = A.submatrix(k, k, m - k, 1);
			
			// Skip if column is effectively zero, we have nothing to do
			if(a.norm() > 0.00001) {
				 // Create first basis vector scaled by -||a||
				Matrix e1 = new Matrix(a.getVSize(), 1);
				e1.setEntry(0, 0, add(0, -a.norm()));
				
				// Householder vector v = a + e1
				Matrix v1 = Matrix.add(a, e1);
				
				// Compute Householder reflection: U = I - 2*v*v^T / ||v||^2
				Matrix U = Matrix.multiply(v1, v1.transpose());
				double d = v1.normSquared();
				U = U.scalmul(div(-2.0, d));
				
				U = Matrix.addIdentity(U);
				
				// Apply Householder from left to zero out sub-diagonal entries
				A = Matrix.embSquareMulL(A, U, m);
				
				// Accumulate transformation into Q (P here)
				P = Matrix.embSquareMulR(P, U, m);
			} 
		}
		
		// Return {Q, R} pair
		Matrix[] ret = {P, A};
		return ret;
	}
}
