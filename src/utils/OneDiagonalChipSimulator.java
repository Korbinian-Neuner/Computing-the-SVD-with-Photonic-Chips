package utils;

import main.Logger;


// Simulates a photonic chip that applies a sequence of 2x2 rotations
// arranged along one diagonal. Supports configuring rotations,
// activating them, and applying them from left or right to matrices.
public class OneDiagonalChipSimulator {
	
	// Matrices queued for the next activation step
	private static Matrix[] matrices = new Matrix[1];
	
	// Currently active matrices applied during simulation
	private static Matrix[] activeMatrices = new Matrix[1];
	
	// Sets the number of rotation slots (chip dimension)
	public static void setDimension(int dim) {
		matrices = new Matrix[dim];
		activeMatrices = new Matrix[dim];
	}
	
	// Presets a rotation matrix at a given diagonal index
	public static void presetMatrix(int index, Matrix A) {
		matrices[index] = A;
	}
	
	// Activates the currently preset matrices:
	// moves them into activeMatrices and clears staging area.
	public static void activate() {
		Logger.logOperation("CHIP_CONf");
		activeMatrices = matrices;
		matrices = new Matrix[matrices.length];
	}
	
	// Applies active rotations from the right side of matrix A.
	// Each column is processed independently.
	public static Matrix shineThroughRight(Matrix A) {
		int m = A.getVSize();
		int n = A.getHSize();
		
		Matrix ret = new Matrix(m, n);
		
		// Iterate through columns
		for(int i = 0; i < n; i++) {
			Logger.logOperation("CHIP_MULT");
			
			// Copy column from input matrix
			for(int j = 0; j < m; j++) {
				ret.setEntry(j, i, A.getEntry(j, i));
			}
			
			// Apply rotations top-down along diagonal
			for(int j = 0; j < m - 1; j++) {
				Matrix r = activeMatrices[j];
				double d1 = ret.getEntry(j, i);
				double d2 = ret.getEntry(j + 1, i);
				if(r == null) continue;
				
				// Extract rotation coefficients
				double r00 = r.getEntry(0, 0);
				double r01 = r.getEntry(0, 1);
				double r10 = r.getEntry(1, 0);
				double r11 = r.getEntry(1, 1);
				
				// Apply 2x2 rotation to adjacent entries
				ret.setEntry(j, i, r00 * d1 + r01 * d2);
				ret.setEntry(j + 1, i, r10 * d1 + r11 * d2);
			}
		}
		
		return ret;
	}
	
	// Applies active rotations to A but like the chip is in the flipped configuration.
	// Similar to shineThroughRight but rotations are applied bottom-up.
	public static Matrix shineThroughLeft(Matrix A) {
		int m = A.getVSize();
		int n = A.getHSize();
		
		Matrix ret = new Matrix(m, n);
		
		// Iterate through columns
		for(int i = 0; i < n; i++) {
			Logger.logOperation("CHIP_MULT");
			
			// Copy column from input matrix
			for(int j = 0; j < m; j++) {
				ret.setEntry(j, i, A.getEntry(j, i));
			}
			
			// Apply rotations bottom-up along diagonal
			for(int j = m - 2; j >= 0; j--) {
				Matrix r = activeMatrices[j];
				double d1 = ret.getEntry(j, i);
				double d2 = ret.getEntry(j + 1, i);
				if(r == null) continue;
				
				// Extract rotation coefficients
				double r00 = r.getEntry(0, 0);
				double r01 = r.getEntry(0, 1);
				double r10 = r.getEntry(1, 0);
				double r11 = r.getEntry(1, 1);
				
				// Apply 2x2 rotation to adjacent entries
				ret.setEntry(j, i, r00 * d1 + r01 * d2);
				ret.setEntry(j + 1, i, r10 * d1 + r11 * d2);
			}
		}
		
		return ret;
	}
	
	// Clears both staged and active matrices while keeping dimension
	public static void reset() {
		matrices = new Matrix[matrices.length];
		activeMatrices = new Matrix[matrices.length];
	}
}
