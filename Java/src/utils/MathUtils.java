package utils;

import java.math.BigInteger;

import main.Logger;

public class MathUtils {
	public static boolean isParallelized = false;
	
	public static double add(double a, double b) {
		Logger.logOperation("ADD");
		return a + b;
	}
	
	public static double mult(double a, double b) {
		Logger.logOperation("MUL");
		return a * b;
	}
	
	public static double div(double a, double b) {
		Logger.logOperation("DIV");
		return a / b;
	}
	
	public static double sqrt(double a) {
		Logger.logOperation("SQR");
		return Math.sqrt(a);
	}
	
	public static double sign(double a) {
		Logger.logOperation("SIG");
		return Math.signum(a);
	}
	
	public static double abs(double a) {
		Logger.logOperation("ABS");
		return Math.abs(a);
	}
	
	public static double gadd(double a, double b) {
		if(isParallelized) {
			Logger.logOperation("GPU_ADD");
			return a + b;
		}
		return add(a, b);
	}
	
	public static double gmult(double a, double b) {
		if(isParallelized) {
			Logger.logOperation("GPU_MUL");
			return a * b;
		}
		return mult(a, b);
	}
	
	// Converts a 3D integer tensor into polynomial coefficient form along all three axes.
	// s1, s2, s3 define shifts for the sampling positions in each dimension.
	public static Rational[][][] toPol(int[][][] d, int s1, int s2, int s3) {
		// Build power matrix for first dimension: (i+s1)^j
		Rational[][] powers1 = new Rational[d.length][d.length];
		for(int i = 0; i < powers1.length; i++) {
			for(int j = 0; j < powers1[0].length; j++) {
				powers1[i][j] = new Rational(i + s1).pow(j);
			}
		}
		
		// Build power matrix for second dimension: (i+s2)^j
		Rational[][] powers2 = new Rational[d[0].length][d[0].length];
		for(int i = 0; i < powers2.length; i++) {
			for(int j = 0; j < powers2[0].length; j++) {
				powers2[i][j] = new Rational(i + s2).pow(j);
			}
		}
		
		// Build power matrix for third dimension: (i+s3)^j
		Rational[][] powers3 = new Rational[d[0][0].length][d[0][0].length];
		for(int i = 0; i < powers3.length; i++) {
			for(int j = 0; j < powers3[0].length; j++) {
				powers3[i][j] = new Rational(i + s3).pow(j);
			}
		}
		
		// Invert the power matrices to later convert values into polynomial coefficients
		Rational[][] inv1 = invert(powers1);
		Rational[][] inv2 = invert(powers2);
		Rational[][] inv3 = invert(powers3);
		
		// Result tensor holding transformed polynomial coefficients
		Rational[][][] ret = new Rational[d.length][d[0].length][d[0][0].length];
		
		
		// --- Transform along third dimension ---
		// For each (i,j) slice, treat values along k as a vector and convert
		// from sampled values to polynomial coefficients using inv3
		for(int i = 0; i < ret.length; i++) {
			for(int j = 0; j < ret[0].length; j++) {
				Rational[] v3 = new Rational[ret[0][0].length];
				
				// Copy integer values into Rational vector
				for(int k = 0; k < ret[0][0].length; k++) {
					int dd = d[i][j][k];
					v3[k] = new Rational(dd);
				}
				// Apply inverse Vandermonde transform
				v3 = multiply(inv3, v3);
				
				// Store transformed coefficients
				for(int k = 0; k < ret[0][0].length; k++) {
					ret[i][j][k] = v3[k];
				}
			}
		}
		
		// --- Transform along first dimension ---
		// For each (i,j) plane orthogonal to axis 1, convert vectors along axis 1
		for(int i = 0; i < ret[0].length; i++) {
			for(int j = 0; j < ret[0][0].length; j++) {
				Rational[] v1 = new Rational[ret.length];
				
				// Extract vector along dimension 1
				for(int k = 0; k < ret.length; k++) {
					v1[k] = ret[k][i][j];
				}
				
				// Apply inverse transform
				v1 = multiply(inv1, v1);
				
				// Write back coefficients
				for(int k = 0; k < ret.length; k++) {
					ret[k][i][j] = v1[k];
				}
			}
		}
		
		// --- Transform along second dimension ---
		// For each (i,j) plane orthogonal to axis 2, convert vectors along axis 2
		for(int i = 0; i < ret[0][0].length; i++) {
			for(int j = 0; j < ret.length; j++) {
				Rational[] v2 = new Rational[ret[0].length];
				
				// Extract vector along dimension 2
				for(int k = 0; k < ret[0].length; k++) {
					v2[k] = ret[j][k][i];
				}
				
				// Apply inverse transform
				v2 = multiply(inv2, v2);
				
				// Write back coefficients
				for(int k = 0; k < ret[0].length; k++) {
					ret[j][k][i] = v2[k];
				}
			}
		}
		
		return ret;
	}
	
	
	
	// Inverts a Rationalmatrix
	public static Rational[][] invert(Rational[][] A) {
	    int n = A.length;
	    Rational[][] aug = new Rational[n][2*n];

	    // Construct [A | I]
	    for (int i = 0; i < n; i++) {
	        for (int j = 0; j < n; j++) {
	            aug[i][j] = A[i][j];
	        }
	        for (int j = n; j < 2*n; j++) {
	            aug[i][j] = (j - n == i) ? new Rational(1) : new Rational(0);
	        }
	    }

	    for (int i = 0; i < n; i++) {
	        Rational pivot = aug[i][i];
	        if (pivot.isZero())
	            throw new ArithmeticException("Matrix is singular");

	        // Normalize pivot row
	        for (int j = 0; j < 2*n; j++)
	            aug[i][j] = aug[i][j].divide(pivot);

	        // Eliminate other rows
	        for (int r = 0; r < n; r++) {
	            if (r != i) {
	                Rational factor = aug[r][i];
	                for (int c = 0; c < 2*n; c++) {
	                    aug[r][c] = aug[r][c].subtract(factor.multiply(aug[i][c]));
	                }
	            }
	        }
	    }

	    // Extract inverse
	    Rational[][] inv = new Rational[n][n];
	    for (int i = 0; i < n; i++) {
	        System.arraycopy(aug[i], n, inv[i], 0, n);
	    }

	    return inv;
	}
	
	// Rational Matrix Vector multiplication
	public static Rational[] multiply(Rational[][] A, Rational[] v) {
	    int n = A.length;
	    if (A[0].length != v.length)
	        throw new IllegalArgumentException("Matrix column count must match vector length");

	    Rational[] result = new Rational[n];

	    for (int i = 0; i < n; i++) {
	        Rational sum = new Rational(0);

	        for (int j = 0; j < v.length; j++) {
	            sum = sum.add(A[i][j].multiply(v[j]));
	        }

	        result[i] = sum;
	    }

	    return result;
	}

}
