package main;

import static utils.MathUtils.*;
import utils.Matrix;

public class SVD {
	
	// Computes the SVD of matrix A using bidiagonalization + QR iteration.
	// Returns {U, B, V} where B approximates the diagonal singular value matrix.
	public static Matrix[] calcSVD(Matrix A, int iterations) {
		// Ensure matrix is tall (m >= n) by transposing if necessary
		if(A.getVSize() < A.getHSize()) {
			A = A.transpose();
		}
		
		// First reduce A to bidiagonal form
		Matrix[] bidiag = bidiagonalize(A);
		Logger.nextSection("Bidiagonalization");
		
		// Perform iterative QR algorithm on bidiagonal matrix
		Matrix[] iterated = qr_iterate(bidiag, iterations);
		Logger.nextSection("QR-Iteration");
		return iterated;
	}
	
	// Reduces A to bidiagonal form using Householder reflections.
	// Returns {P, B, Q} such that P*A*Q = B (bidiagonal).
	public static Matrix[] bidiagonalize(Matrix A) {
		int m = A.getVSize();
		int n = A.getHSize();
		
		
		// Accumulated orthogonal transformations
		Matrix P = Matrix.createIdentity(m);
		Matrix Q = Matrix.createIdentity(n);
		
		for(int k = 0; k < n; k++) {
			// Extract column subvector below diagonal
			Matrix a = A.submatrix(k, k, m - k, 1);
			if(a.norm() > 0.00001) {
				// Construct Householder vector
				Matrix e1 = new Matrix(a.getVSize(), 1);
				e1.setEntry(0, 0, add(0, -a.norm()));
				Matrix v1 = Matrix.add(a, e1);
				
				// Build Householder matrix U
				Matrix U = Matrix.multiply(v1, v1.transpose());
				double d = v1.normSquared();
				U = U.scalmul(div(-2.0, d));
				
				if(d < 0.000001) break;
				
				U = Matrix.addIdentity(U);
				
				// Apply from left to zero out below-diagonal entries
				A = Matrix.embSquareMulL(A, U, m);
				P = Matrix.embSquareMulR(P, U, m);
			}
			
			// Stop here if last column, added so it can handle m>n and n>m
			if(k == n-1) break;
			if(k == n-2) continue;
			
			// Extract row subvector to the right of superdiagonal
			Matrix b = A.submatrix(k, k + 1, 1, n - k - 1);
			if(b.norm() > 0.00001) {
				Matrix e1 = new Matrix(1, b.getHSize());
				e1.setEntry(0, 0, add(0, -b.norm()));
				Matrix v2 = Matrix.add(b, e1);
				
				// Build Householder matrix V
				Matrix V = Matrix.multiply(v2.transpose(), v2);
				double d = v2.normSquared();
				V = V.scalmul(div(-2.0, d));
				
				V = Matrix.addIdentity(V);
				
				// Apply from right to zero out entries
				A = Matrix.embSquareMulR(A, V, n);
				Q = Matrix.embSquareMulL(Q, V, n);
				
				
			}
		}
		
		
		Matrix[] ret = {P, A, Q};
		return ret;
	}
	
	// Performs implicit QR iteration on bidiagonal matrix to approximate singular values.
	// Accumulates left and right rotations into UAcc and VAcc.
	public static Matrix[] qr_iterate(Matrix[] bid, int iterations) {
		Matrix A = bid[1];
		
		int m = A.getVSize();
		int n = A.getHSize();
		
		Matrix UAcc = bid[0];
		Matrix VAcc = bid[2];
		
		for(int iter = 0; iter < iterations; iter++) {
			// Compute Wilkinson-like shift
			double shift = calc_shift(A);
			
			// Initial right Givens rotation parameter
			double r_1 = mult(A.getEntry(0, 1), A.getEntry(0, 0));
			r_1 = div(r_1, add(mult(A.getEntry(0, 0), A.getEntry(0, 0)), -shift));
			
			Matrix R_1 = Matrix.createGivensRotation(r_1, 2, 0, 1);
			
			
			// Apply first rotation manually to leading 2x2 block
			Logger.setcheckpoint();
			double d1 = gadd(gmult(A.getEntry(0, 0), R_1.getEntry(0, 0)), gmult(A.getEntry(0, 1), R_1.getEntry(1, 0)));
			Logger.gotoCheckpoint();
			double d2 = gadd(gmult(A.getEntry(1, 0), R_1.getEntry(0, 0)), gmult(A.getEntry(1, 1), R_1.getEntry(1, 0)));
			Logger.gotoCheckpoint();
			double d3 = gadd(gmult(A.getEntry(0, 0), R_1.getEntry(0, 1)), gmult(A.getEntry(0, 1), R_1.getEntry(1, 1)));
			Logger.gotoCheckpoint();
			double d4 = gadd(gmult(A.getEntry(1, 0), R_1.getEntry(0, 1)), gmult(A.getEntry(1, 1), R_1.getEntry(1, 1)));
			double d5;
			double d6;
			
			A.setEntry(0, 0, d1);
			A.setEntry(1, 0, d2);
			A.setEntry(0, 1, d3);
			A.setEntry(1, 1, d4);
			
			// Accumulate rotation into V
			VAcc = Matrix.applyGivensRotationLeft(VAcc, add(0, -r_1), n, 0, 1, false);
			
			// Bulge chasing through bidiagonal matrix
			for(int k = 0; k < n - 2; k++) {
				double l_k = add(0, -div(A.getEntry(k + 1, k), A.getEntry(k, k)));
				
				Matrix L_k = Matrix.createGivensRotation(l_k, 2, 0, 1);
				
				// Apply left rotation to local 2x3 block
				Logger.setcheckpoint();
				d1 = gadd(gmult(A.getEntry(k, k), L_k.getEntry(0, 0)), gmult(A.getEntry(k+1, k), L_k.getEntry(0, 1)));
				Logger.gotoCheckpoint();
				d2 = gadd(gmult(A.getEntry(k, k), L_k.getEntry(1, 0)), gmult(A.getEntry(k+1, k), L_k.getEntry(1, 1)));
				Logger.gotoCheckpoint();
				d3 = gadd(gmult(A.getEntry(k, k+1), L_k.getEntry(0, 0)), gmult(A.getEntry(k+1, k+1), L_k.getEntry(0, 1)));
				Logger.gotoCheckpoint();
				d4 = gadd(gmult(A.getEntry(k, k+1), L_k.getEntry(1, 0)), gmult(A.getEntry(k+1, k+1), L_k.getEntry(1, 1)));
				Logger.gotoCheckpoint();
				d5 = gadd(gmult(A.getEntry(k, k+2), L_k.getEntry(0, 0)), gmult(A.getEntry(k+1, k+2), L_k.getEntry(0, 1)));
				Logger.gotoCheckpoint();
				d6 = gadd(gmult(A.getEntry(k, k+2), L_k.getEntry(1, 0)), gmult(A.getEntry(k+1, k+2), L_k.getEntry(1, 1)));
				
				A.setEntry(k, k, d1);
				A.setEntry(k+1, k, d2);
				
				A.setEntry(k, k+1, d3);
				A.setEntry(k+1, k+1, d4);
				
				A.setEntry(k, k+2, d5);
				A.setEntry(k+1, k+2, d6);
				
				
				UAcc = Matrix.applyGivensRotationRight(UAcc, add(0, -l_k), m, k, k+1, false);
				
				
				// Right rotation to chase bulge
				double r_kp1 = div(A.getEntry(k, k + 2), A.getEntry(k, k + 1));
				
				Matrix R_kp1 = Matrix.createGivensRotation(r_kp1, 2, 0, 1);	
				
				Logger.setcheckpoint();
				d1 = gadd(gmult(A.getEntry(k, k+1), R_kp1.getEntry(0, 0)), gmult(A.getEntry(k, k+2), R_kp1.getEntry(1, 0)));
				Logger.gotoCheckpoint();
				d2 = gadd(gmult(A.getEntry(k, k+1), R_kp1.getEntry(0, 1)), gmult(A.getEntry(k, k+2), R_kp1.getEntry(1, 1)));
				Logger.gotoCheckpoint();
				d3 = gadd(gmult(A.getEntry(k+1, k+1), R_kp1.getEntry(0, 0)), gmult(A.getEntry(k+1, k+2), R_kp1.getEntry(1, 0)));
				Logger.gotoCheckpoint();
				d4 = gadd(gmult(A.getEntry(k+1, k+1), R_kp1.getEntry(0, 1)), gmult(A.getEntry(k+1, k+2), R_kp1.getEntry(1, 1)));
				Logger.gotoCheckpoint();
				d5 = gadd(gmult(A.getEntry(k+2, k+1), R_kp1.getEntry(0, 0)), gmult(A.getEntry(k+2, k+2), R_kp1.getEntry(1, 0)));
				Logger.gotoCheckpoint();
				d6 = gadd(gmult(A.getEntry(k+2, k+1), R_kp1.getEntry(0, 1)), gmult(A.getEntry(k+2, k+2), R_kp1.getEntry(1, 1)));
				
				A.setEntry(k, k+1, d1);
				A.setEntry(k, k+2, d2);
				
				A.setEntry(k+1, k+1, d3);
				A.setEntry(k+1, k+2, d4);
				
				A.setEntry(k+2, k+1, d5);
				A.setEntry(k+2, k+2, d6);
				
				
				VAcc = Matrix.applyGivensRotationLeft(VAcc, add(0, -r_kp1), n, k+1, k+2, false);
			}
			
			// Final left rotation for last block
			double ll =  add(0, -div(A.getEntry(n - 1, n - 2), A.getEntry(n - 2, n - 2)));
			Matrix L_l = Matrix.createGivensRotation(ll, 2, 0, 1);
			
			Logger.setcheckpoint();
			d1 = gadd(gmult(A.getEntry(n-2, n-2), L_l.getEntry(0, 0)), gmult(A.getEntry(n-1, n-2), L_l.getEntry(0, 1)));
			Logger.gotoCheckpoint();
			d2 = gadd(gmult(A.getEntry(n-2, n-2), L_l.getEntry(1, 0)), gmult(A.getEntry(n-1, n-2), L_l.getEntry(1, 1)));
			Logger.gotoCheckpoint();
			d3 = gadd(gmult(A.getEntry(n-2, n-1), L_l.getEntry(0, 0)), gmult(A.getEntry(n-1, n-1), L_l.getEntry(0, 1)));
			Logger.gotoCheckpoint();
			d4 = gadd(gmult(A.getEntry(n-2, n-1), L_l.getEntry(1, 0)), gmult(A.getEntry(n-1, n-1), L_l.getEntry(1, 1)));
			
			A.setEntry(n-2, n-2, d1);
			A.setEntry(n-1, n-2, d2);
			
			A.setEntry(n-2, n-1, d3);
			A.setEntry(n-1, n-1, d4);
			
			UAcc = Matrix.applyGivensRotationRight(UAcc, add(0, -ll), m, n-2, n-1, false);
			
		}
		
		Matrix[] ret = {UAcc, A, VAcc};
		return ret;
	} 
	
	// Computes the Wilkinson-shift based on bottom-right 2x2 block.
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
