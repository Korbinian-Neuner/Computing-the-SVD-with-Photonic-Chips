package main;

import utils.Matrix;
import utils.OneDiagonalChipSimulator;
import static utils.MathUtils.*;

public class PhotonicQRSVD {
	
	public static Matrix[] calcSVD(Matrix A, int iterations) {
		// Ensure matrix is tall (m >= n); transpose if necessary
		if(A.getVSize() < A.getHSize()) {
			A = A.transpose();
		}
		
		int m = A.getVSize();
		int n = A.getHSize();
		
		// Initialize U and V as identity matrices
		Matrix U = Matrix.createIdentity(m);
		Matrix V = Matrix.createIdentity(n);
		
		// Configure simulator dimension
		OneDiagonalChipSimulator.setDimension(m - 1);
		
		// Main QR-style SVD iteration loop
		for(int it = 0; it < iterations; it++) {
			m = A.getVSize();
			n = A.getHSize();
			
			// Left reduction phase (apply rotations column-wise)
			for(int k = 0; k < n; k++) {
				// Start from bottom element of column k
				double temp = A.getEntry(m-1, k);
				
				// Go upwards zeroing out everything using Givens rotations
				for(int l = m - 1; l > k; l--) {
					// Compute rotation parameter r
					double r_kl = div(add(0, -temp), A.getEntry(l-1, k));
					
					// Handle near-zero edge case
					if(Math.abs(temp) < 0.0000000001 && Math.abs(A.getEntry(l-1, k)) < 0.0000000001) r_kl = 1.0;
					
					// Create 2x2 Givens rotation
					Matrix g = Matrix.createGivensRotation(r_kl, 2, 0, 1);
					
					// Store rotation into simulator pipeline
					OneDiagonalChipSimulator.presetMatrix(l-1, g);
					
					// Update temp to reflect rotation effect
					temp = div(add(A.getEntry(l-1, k), -mult(r_kl, temp)), sqrt(add(1, mult(r_kl, r_kl))));
				}
				
				// Apply rotations to A and accumulate into U
				OneDiagonalChipSimulator.activate();
				A = OneDiagonalChipSimulator.shineThroughLeft(A);
				U = OneDiagonalChipSimulator.shineThroughLeft(U.transpose()).transpose();
			}
			
			A = A.transpose();
			
			// Right reduction phase (now again left rotations due to transpose)
			for(int k = 0; k < m; k++) {
				// Start from bottom element
				double temp = A.getEntry(n-1, k);
				
				// Go upwards zeroing out everything using Givens rotations
				for(int l = n - 1; l > k; l--) {
					// Compute rotation parameter r
					double r_kl = div(add(0, -temp), A.getEntry(l-1, k));
					
					// Handle near-zero edge case
					if(Math.abs(temp) < 0.0000000001 && Math.abs(A.getEntry(l-1, k)) < 0.0000000001) r_kl = 1.0;
					
					// Create 2x2 Givens rotation
					Matrix g = Matrix.createGivensRotation(r_kl, 2, 0, 1);
					
					// Store rotation into simulator pipeline
					OneDiagonalChipSimulator.presetMatrix(l-1, g);
					
					// Update temp to reflect rotation effect
					temp = div(add(A.getEntry(l-1, k), -mult(r_kl, temp)), sqrt(add(1, mult(r_kl, r_kl))));
				}
				
				// Apply rotations to A and accumulate into V
				OneDiagonalChipSimulator.activate();
				A = OneDiagonalChipSimulator.shineThroughLeft(A);
				V = OneDiagonalChipSimulator.shineThroughLeft(V);
			}
			
			// Restore original orientation
			A = A.transpose();
		}
		
		// Return U, Σ (stored in A), and V
		Matrix[] ret = {U, A, V};
		return ret;
	}

}
