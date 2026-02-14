package main;

import utils.Matrix;
import utils.OneDiagonalChipSimulator;
import static utils.MathUtils.*;

//Implements a photonic-style Singular Value Decomposition (SVD)
//using bidiagonalization followed by QR iteration.
//Uses OneDiagonalChipSimulator to simulate sequences of Givens rotations.
public class PhotonicSVD {
	
	// High-level SVD pipeline:
	// 1. Ensure matrix is tall
	// 2. Bidiagonalize
	// 3. Perform QR iterations
	// 4. Construct final U and V unitaries
	public static Matrix[] calcSVD(Matrix A, int iterations) {
		if(A.getVSize() < A.getHSize()) {
			A = A.transpose();
		}
		// Step 1: Reduce matrix to bidiagonal form
		Matrix[] bidiag = bidiagonalize(A);
		Logger.nextSection("Bidiagonalization");
		
		// Step 2: Perform iterative QR-based diagonalization
		Matrix[] ret= qr_iterate(bidiag, iterations);
		Logger.nextSection("QR-Iteration");
		return ret;
	}
	
	// Reduces matrix A into bidiagonal form using sequences of Givens rotations.
	// Returns {P, B, Q} where B is bidiagonal and A = P * B * Q.
	public static Matrix[] bidiagonalize(Matrix A) {
		int m = A.getVSize();
		int n = A.getHSize();
		
		// Accumulated left and right transformations
		Matrix P = Matrix.createIdentity(m);
		Matrix Q = Matrix.createIdentity(n);
		
		// Configure photonic chip simulator
		OneDiagonalChipSimulator.setDimension(m - 1);
		
		// Main bidiagonalization loop
		for(int k = 0; k < n; k++) {
			
			// --- Left rotations: zero out subdiagonal entries ---
			double temp = A.getEntry(m-1, k);
			Matrix t = A;
			
			for(int l = m - 1; l > k; l--) {
				double r_kl = div(add(0, -temp), A.getEntry(l-1, k));
				Matrix g = Matrix.createGivensRotation(r_kl, 2, 0, 1);
				OneDiagonalChipSimulator.presetMatrix(l-1, g);
				
				// Update running value for next rotation
				temp = div(add(A.getEntry(l-1, k), -mult(r_kl, temp)), sqrt(add(1, mult(r_kl, r_kl))));
			}
			
			// Apply accumulated left rotations
			OneDiagonalChipSimulator.activate();
			A = OneDiagonalChipSimulator.shineThroughLeft(A);
			P = OneDiagonalChipSimulator.shineThroughLeft(P.transpose()).transpose();
			
			// --- Right rotations: zero out superdiagonal entries ---
			temp = A.getEntry(k, n - 1);
			for(int l = n - 1; l > k + 1; l--) {
				double s_kl = div(temp, A.getEntry(k, l-1));
				Matrix g = Matrix.createGivensRotation(add(0, -s_kl), 2, 0, 1);
				OneDiagonalChipSimulator.presetMatrix(l-1, g);
				
				temp = div(add(A.getEntry(k, l - 1), mult(s_kl, temp)), sqrt(add(1, mult(s_kl, s_kl))));
			}
			
			// Apply accumulated right rotations
			OneDiagonalChipSimulator.activate();
			A = OneDiagonalChipSimulator.shineThroughLeft(A.transpose()).transpose();
			Q = OneDiagonalChipSimulator.shineThroughLeft(Q);
			
			
		}
		
		Matrix[] ret = {P, A, Q};
		return ret;
	}
	
	// Performs iterative QR-like sweeps on a bidiagonal matrix
	// to approach diagonal (singular value) form.
	// Returns the completed SVD
	public static Matrix[] qr_iterate(Matrix[] bidiag, int iterations) {
		Matrix A = bidiag[1];
		
		int m = A.getVSize();
		int n = A.getHSize();
		
		// Clean very small values to zero for numerical stability
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				if(Math.abs(A.getEntry(i, j)) < 0.0000001) {
					A.setEntry(i, j, 0);
				}
			}
		}
		
		// Accumulate left and right singular vector transforms
		Matrix UAcc = bidiag[0];
		Matrix VAcc = bidiag[2];
		
		// Main QR iteration loop
		for(int iter = 0; iter < iterations; iter++) {
			// Compute Wilkinson-shift
			double shift = calc_shift(A);
			
			// Initial right Givens rotation parameter
			double r_1 = mult(A.getEntry(0, 1), A.getEntry(0, 0));
			r_1 = div(r_1, add(mult(A.getEntry(0, 0), A.getEntry(0, 0)), shift));
			
			Matrix[] lrs = new Matrix[n-1]; // Left rotations
			Matrix[] rrs = new Matrix[n-1]; // Right rotations
			
			Matrix R_1 = Matrix.createGivensRotation(r_1, 2, 0, 1);
			
			rrs[0] = R_1;
			
			// Manually apply Givens-rotation
			double d1 = add(mult(A.getEntry(0, 0), R_1.getEntry(0, 0)), mult(A.getEntry(0, 1), -R_1.getEntry(1, 0)));
			double d2 = add(mult(A.getEntry(1, 0), R_1.getEntry(0, 0)), mult(A.getEntry(1, 1), -R_1.getEntry(1, 0)));
			double d3 = add(mult(A.getEntry(0, 0), -R_1.getEntry(0, 1)), mult(A.getEntry(0, 1), R_1.getEntry(1, 1)));
			double d4 = add(mult(A.getEntry(1, 0), -R_1.getEntry(0, 1)), mult(A.getEntry(1, 1), R_1.getEntry(1, 1)));
			double d5 = 0.0;
			double d6 = 0.0;
			
			A.setEntry(0, 0, d1);
			A.setEntry(1, 0, d2);
			A.setEntry(0, 1, d3);
			A.setEntry(1, 1, d4);
			
			// Bulge chasing through matrix
			for(int k = 0; k < n - 2; k++) {
				double l_k = add(0, -div(A.getEntry(k + 1, k), A.getEntry(k, k)));
				Matrix L_k = Matrix.createGivensRotation(l_k, 2, 0, 1);
				//A = Matrix.multiply(L_k, A);
				//UAcc = Matrix.multiply(UAcc, L_k.transpose());
				lrs[k] = L_k;
				
				// Manually apply Givens-rotation
				d1 = add(mult(A.getEntry(k, k), L_k.getEntry(0, 0)), mult(A.getEntry(k+1, k), L_k.getEntry(0, 1)));
				d2 = add(mult(A.getEntry(k, k), L_k.getEntry(1, 0)), mult(A.getEntry(k+1, k), L_k.getEntry(1, 1)));
				d3 = add(mult(A.getEntry(k, k+1), L_k.getEntry(0, 0)), mult(A.getEntry(k+1, k+1), L_k.getEntry(0, 1)));
				d4 = add(mult(A.getEntry(k, k+1), L_k.getEntry(1, 0)), mult(A.getEntry(k+1, k+1), L_k.getEntry(1, 1)));
				d5 = add(mult(A.getEntry(k, k+2), L_k.getEntry(0, 0)), mult(A.getEntry(k+1, k+2), L_k.getEntry(0, 1)));
				d6 = add(mult(A.getEntry(k, k+2), L_k.getEntry(1, 0)), mult(A.getEntry(k+1, k+2), L_k.getEntry(1, 1)));
				
				A.setEntry(k, k, d1);
				A.setEntry(k+1, k, d2);
				
				A.setEntry(k, k+1, d3);
				A.setEntry(k+1, k+1, d4);
				
				A.setEntry(k, k+2, d5);
				A.setEntry(k+1, k+2, d6);
				
				
				// Right rotation to chase bulge
				double r_kp1 = add(0, -div(A.getEntry(k, k + 2), A.getEntry(k, k + 1)));
				Matrix R_kp1 = Matrix.createGivensRotation(r_kp1, 2, 0, 1);
				rrs[k+1] = R_kp1;
				
				d1 = add(mult(A.getEntry(k, k+1), R_kp1.getEntry(0, 0)), mult(A.getEntry(k, k+2), -R_kp1.getEntry(1, 0)));
				d2 = add(mult(A.getEntry(k, k+1), -R_kp1.getEntry(0, 1)), mult(A.getEntry(k, k+2), R_kp1.getEntry(1, 1)));
				d3 = add(mult(A.getEntry(k+1, k+1), R_kp1.getEntry(0, 0)), mult(A.getEntry(k+1, k+2), -R_kp1.getEntry(1, 0)));
				d4 = add(mult(A.getEntry(k+1, k+1), -R_kp1.getEntry(0, 1)), mult(A.getEntry(k+1, k+2), R_kp1.getEntry(1, 1)));
				d5 = add(mult(A.getEntry(k+2, k+1), R_kp1.getEntry(0, 0)), mult(A.getEntry(k+2, k+2), -R_kp1.getEntry(1, 0)));
				d6 = add(mult(A.getEntry(k+2, k+1), -R_kp1.getEntry(0, 1)), mult(A.getEntry(k+2, k+2), R_kp1.getEntry(1, 1)));
				
				A.setEntry(k, k+1, d1);
				A.setEntry(k, k+2, d2);
				
				A.setEntry(k+1, k+1, d3);
				A.setEntry(k+1, k+2, d4);
				
				A.setEntry(k+2, k+1, d5);
				A.setEntry(k+2, k+2, d6);
				
			}
			
			// Final left rotation for last block
			double ll =  add(0, -div(A.getEntry(n - 1, n - 2), A.getEntry(n - 2, n - 2)));
			
			Matrix L_l = Matrix.createGivensRotation(ll, 2, 0, 1);
			lrs[n-2] = L_l;
			
			d1 = add(mult(A.getEntry(n-2, n-2), L_l.getEntry(0, 0)), mult(A.getEntry(n-1, n-2), L_l.getEntry(0, 1)));
			d2 = add(mult(A.getEntry(n-2, n-2), L_l.getEntry(1, 0)), mult(A.getEntry(n-1, n-2), L_l.getEntry(1, 1)));
			d3 = add(mult(A.getEntry(n-2, n-1), L_l.getEntry(0, 0)), mult(A.getEntry(n-1, n-1), L_l.getEntry(0, 1)));
			d4 = add(mult(A.getEntry(n-2, n-1), L_l.getEntry(1, 0)), mult(A.getEntry(n-1, n-1), L_l.getEntry(1, 1)));
			
			A.setEntry(n-2, n-2, d1);
			A.setEntry(n-1, n-2, d2);
			
			A.setEntry(n-2, n-1, d3);
			A.setEntry(n-1, n-1, d4);
			
			// Apply accumulated rotations via photonic simulator
			
			OneDiagonalChipSimulator.reset();
			OneDiagonalChipSimulator.setDimension(m);
			for(int i = 0; i < n-1; i++) {
				OneDiagonalChipSimulator.presetMatrix(i, rrs[i]);
			}
			OneDiagonalChipSimulator.activate();
			VAcc = OneDiagonalChipSimulator.shineThroughRight(VAcc);
			OneDiagonalChipSimulator.reset();
			OneDiagonalChipSimulator.setDimension(m);
			for(int i = 0; i < n-1; i++) {
				OneDiagonalChipSimulator.presetMatrix(i, lrs[i]);
			}
			OneDiagonalChipSimulator.activate();
			UAcc = OneDiagonalChipSimulator.shineThroughRight(UAcc.transpose()).transpose();
			
			
			// STILL MISSING: The extra cases needed for convergence
			// Not implemented as they are not part of the operation counting
		}
		
		Matrix[] ret = {UAcc, A, VAcc};
		return ret;
	}
	
	// Computes the Wilkinson-shift
	public static double calc_shift(Matrix A) {
		int n = A.getHSize();
		
		double a = add(mult(A.getEntry(n - 3, n - 2), A.getEntry(n - 3, n - 2)), mult(A.getEntry(n - 2, n - 2), A.getEntry(n - 2, n - 2)));
		double b = mult(A.getEntry(n - 2, n - 1), A.getEntry(n - 2, n - 2));
		double c = add(mult(A.getEntry(n - 2, n - 1), A.getEntry(n - 2, n - 1)), mult(A.getEntry(n - 1, n - 1), A.getEntry(n - 1, n - 1)));
		
		double d = add(a, -c);
		double r = sqrt(add(mult(d, d), mult(b, b)));
		
		d = add(a, c);
		if(a > c) {
			return div(add(d, -r), 2);
		} else {
			return div(add(d, r), 2);
		}
	}

}
