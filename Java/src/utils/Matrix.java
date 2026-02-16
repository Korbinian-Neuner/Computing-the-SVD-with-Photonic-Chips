package utils;

import java.util.Random;

import main.Logger;
import utils.MathUtils;

public class Matrix {

	// Stores the actual matrix data
	private double[][] contents;
	// Fixed RNG used for reproducible random matrices (default seed 17)
	private static Random rng = new Random(17);
	
	// Creates an empty matrix with given vertical (rows) and horizontal (columns) size
	public Matrix(int vSize, int hSize) {
		contents = new double[vSize][hSize];
	}
	
	// Creates a matrix directly from an existing 2D array (no deep copy!)
	public Matrix(double[][] contents) {
		this.contents = contents;
	}
	
	// Formatted string representation of the matrix with fixed-width entries
	@Override
	public String toString() {
		String ret = "";
		for (int i = 0; i < getVSize(); i++) {
			for (int j = 0; j < getHSize(); j++) {
				String con = "" + getEntry(i, j);
				// Treat very small numbers as zero for readability
				if(Math.abs(getEntry(i, j)) < 0.00001) con = "0.0";
				String c = "";
				// Pad / trim entries to fixed width
				for (int k = 0; k < 6; k++) {
					if (k < con.length()) {
						c = c + con.charAt(k);
					} else {
						c = c + " ";
					}
				}
				ret = ret + c + "    ";
			}
			ret = ret + "\n";
		}
		return ret;
	}
	
	
	// Puts the small Matrix A into a bigger matrix, m1 and n1 give position of A00 entry, vSize and hSize give size of the new matrix
	public static Matrix createEmbeddedMatrix(Matrix A, int m1, int n1, int vSize, int hSize) {
		Matrix ret = new Matrix(vSize, hSize);
		for (int i = 0; i < A.getVSize(); i++) {
			for (int j = 0; j < A.getHSize(); j++) {
				ret.setEntry(i + m1, j + n1, A.getEntry(i, j));
			}
		}

		return ret;
	}
	
	// Creates a random matrix of specified Size for testing purposes
	public static Matrix createRandomMatrix(int vSize, int hSize) {
		Matrix ret = new Matrix(vSize, hSize);
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				ret.setEntry(i, j, rng.nextDouble());
			}
		}

		return ret;
	}
	
	// Creates a random matrix of specified Size for testing purposes using a seed
	public static Matrix createRandomMatrix(int vSize, int hSize, long seed) {
		Random rng = new Random(seed);
		Matrix ret = new Matrix(vSize, hSize);
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				ret.setEntry(i, j, rng.nextDouble());
			}
		}

		return ret;
	}
	
	// Creates an identity matrix
	public static Matrix createIdentity(int size) {
		Matrix ret = new Matrix(size, size);
		for (int i = 0; i < size; i++) {
			ret.setEntry(i, i, 1);
		}
		return ret;
	}
	
	// Creates a Givens rotation matrix embedded in a larger identity matrix.
	// r = tan(theta) parametrization, l and k are the rotated indices.
	public static Matrix createGivensRotation(double r, int dim, int l, int k) {
		if (l >= k || l >= dim || k > dim) {
			throw new IllegalArgumentException("Incorrect Arguments for Givens rotation");
		}
		
		// Compute cosine and sine values from r
		double c = MathUtils.div(1, MathUtils.sqrt(MathUtils.add(1, MathUtils.mult(r, r))));
		double s = MathUtils.mult(c, r);

		Matrix ret = createIdentity(dim);
		// Modify identity entries to embed rotation
		ret.setEntry(l, l, c);
		ret.setEntry(k, l, s);
		ret.setEntry(l, k, MathUtils.add(0, -s));
		ret.setEntry(k, k, c);

		return ret;
	}
	
	// Applies a Givens rotation from the right: A -> A * G
	// Only optionally logs initialization operations
	public static Matrix applyGivensRotationRight(Matrix A, double r, int dim, int l, int k, boolean logInitialisation) {
		Matrix ret = A.copy();
		
		if (l >= k || l >= dim || k > dim) {
			throw new IllegalArgumentException("Incorrect Arguments for Givens rotation");
		}
		
		// Compute rotation parameters
		if(!logInitialisation) Logger.setActive(false);
		double c = MathUtils.div(1, MathUtils.sqrt(MathUtils.add(1, MathUtils.mult(r, r))));
		double s = MathUtils.mult(c, r);
		double ms = MathUtils.add(0, -s);
		if(!logInitialisation) Logger.setActive(true);
		
		// Apply rotation to column pairs
		Logger.setcheckpoint();
		for(int i = 0; i < A.getVSize(); i++) {
			double ail = A.getEntry(i, l);
			double aik = A.getEntry(i, k);
			
			Logger.gotoCheckpoint();
			ret.setEntry(i, l, MathUtils.gadd(MathUtils.gmult(ail, c), MathUtils.gmult(aik, s)));
			
			Logger.gotoCheckpoint();
			ret.setEntry(i, k, MathUtils.gadd(MathUtils.gmult(ail, ms), MathUtils.gmult(aik, c)));
		}
		return ret;
	}
	
	// Same as applyGivensRotationRight(...) but the Givens-Rotation is applied from the left
	public static Matrix applyGivensRotationLeft(Matrix A, double r, int dim, int l, int k, boolean logInitialisation) {
		Matrix ret = A.copy();
		
		if (l >= k || l >= dim || k > dim) {
			throw new IllegalArgumentException("Incorrect Arguments for Givens rotation");
		}
		
		// Compute rotation parameters
		if(!logInitialisation) Logger.setActive(false);
		double c = MathUtils.div(1, MathUtils.sqrt(MathUtils.add(1, MathUtils.mult(r, r))));
		double s = MathUtils.mult(c, r);
		double ms = MathUtils.add(0, -s);
		if(!logInitialisation) Logger.setActive(true);
		
		// Apply rotation to column pairs
		Logger.setcheckpoint();
		for(int i = 0; i < A.getHSize(); i++) {
			double ali = A.getEntry(l, i);
			double aki = A.getEntry(k, i);
			
			Logger.gotoCheckpoint();
			ret.setEntry(l, i, MathUtils.gadd(MathUtils.gmult(ali, c), MathUtils.gmult(aki, ms)));
			
			Logger.gotoCheckpoint();
			ret.setEntry(k, i, MathUtils.gadd(MathUtils.gmult(ali, s), MathUtils.gmult(aki, c)));
			
		}
		return ret;
	}
	
	// Returns a new matrix that is the sum of a and b. Is parallelized for the sake of the Op-counter
	public static Matrix add(Matrix a, Matrix b) {
		if (a.getVSize() != b.getVSize() || a.getHSize() != b.getHSize()) {
			throw new IllegalArgumentException("Tried to add Matrices of different sizes");
		}
		Matrix ret = new Matrix(a.getVSize(), a.getHSize());
		Logger.setIncTimeStep(false);
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				ret.setEntry(i, j, MathUtils.gadd(a.getEntry(i, j), b.getEntry(i, j)));
			}
		}
		Logger.setIncTimeStep(true);
		return ret;
	}
	
	// Returns a new matrix that is the sum of a and the identity. Done this way for operation counting efficiency. Is parallelized for the sake of the Op-counter
	public static Matrix addIdentity(Matrix a) {
		Matrix ret = new Matrix(a.getVSize(), a.getHSize());
		Logger.setIncTimeStep(false);
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				if(i == j) {
					ret.setEntry(i, j, MathUtils.gadd(a.getEntry(i, j), 1.0));
				} else {
					ret.setEntry(i, j, a.getEntry(i, j));
				}
			}
		}
		Logger.setIncTimeStep(true);
		return ret;
	}
	
	// Returns a new matrix that is the product of a and b. Is parallelized for the sake of the Op-counter
	public static Matrix multiply(Matrix a, Matrix b) {
		if (a.getHSize() != b.getVSize()) {
			throw new IllegalArgumentException("Tried to multiply incompatibly Matrices: " + a.getVSize() + "/" + a.getHSize() + " and " + b.getVSize() + "/" + b.getHSize());
		}
		Matrix ret = new Matrix(a.getVSize(), b.getHSize());
		Logger.setcheckpoint();
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				double total = 0;
				Logger.gotoCheckpoint();
				for (int k = 0; k < a.getHSize(); k++) {
					total = MathUtils.gadd(total, MathUtils.gmult(a.getEntry(i, k), b.getEntry(k, j)));
				}
				
				ret.setEntry(i, j, total);
			}
		}

		return ret;
	}
	
	// Multiplication with embedded square matrix from the right
	public static Matrix embSquareMulR(Matrix a, Matrix emb, int dim) {
		if (a.getHSize() != dim) {
			throw new IllegalArgumentException("Tried to multiply incompatibly Embedded Matrices ");
		}
		Matrix ret = new Matrix(a.getVSize(), dim);
		
		// Copy original matrix first
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				ret.setEntry(i, j, a.getEntry(i, j));
			}
		}
		
		
		Logger.setcheckpoint();
		for (int i = 0; i < emb.getHSize(); i++) {
			for (int j = 0; j < ret.getVSize(); j++) {
				double total = 0.0;
				Logger.gotoCheckpoint();
				for(int k = 0; k < emb.getVSize(); k++) {
					total = MathUtils.gadd(total, MathUtils.gmult(a.getEntry(j, k + dim - emb.getVSize()), emb.getEntry(k, i)));
				}
				
				ret.setEntry(j, i + dim - emb.getHSize(), total);
			}
		}

		return ret;
	}
	
	// Multiplication with embedded square matrix from the left implemented via transpose trick
	public static Matrix embSquareMulL(Matrix a, Matrix emb, int dim) {
		return embSquareMulR(a.transpose(), emb.transpose(), dim).transpose();
	}
	
	// Returns transposed matrix
	public Matrix transpose() {
		Matrix ret = new Matrix(getHSize(), getVSize());
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				ret.setEntry(i, j, getEntry(j, i));
			}
		}

		return ret;
	}
	
	// Returns new copy of matrix
	public Matrix copy() {
		Matrix ret = new Matrix(getVSize(), getHSize());
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				ret.setEntry(i, j, getEntry(i, j));
			}
		}

		return ret;
	}
	
	// Scalar multiplication. Returns new matrix
	public Matrix scalmul(double scalar) {
		Matrix ret = new Matrix(getVSize(), getHSize());
		Logger.setIncTimeStep(false);
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				ret.setEntry(i, j, MathUtils.gmult(scalar, getEntry(i, j)));
			}
		}
		Logger.setIncTimeStep(true);
		return ret;
	}
	
	// Number of rows
	public int getVSize() {
		return contents.length;
	}
	
	// Number of columns
	public int getHSize() {
		return contents[0].length;
	}
	
	// Euclidean (Frobenius) norm
	public double norm() {
		return MathUtils.sqrt(normSquared());
	}
	
	// Returns the sum of the squares of all entries. For simplicity its not parallelized with regards to the Op-counter
	public double normSquared() {
		double ret = 0;
		for (int i = 0; i < getVSize(); i++) {
			for (int j = 0; j < getHSize(); j++) {
				ret = MathUtils.add(ret, MathUtils.mult(getEntry(i, j), getEntry(i, j)));
			}
		}
		return ret;
	}
	
	// Extract submatrix starting at (m1,n1)
	public Matrix submatrix(int m1, int n1, int vSize, int hSize) {
		Matrix ret = new Matrix(vSize, hSize);
		for (int i = 0; i < ret.getVSize(); i++) {
			for (int j = 0; j < ret.getHSize(); j++) {
				ret.setEntry(i, j, getEntry(i + m1, j + n1));
			}
		}

		return ret;
	}
	// Access single entry
	public double getEntry(int i, int j) {
		return contents[i][j];
	}
	
	// Modify single entry
	public void setEntry(int i, int j, double newEntry) {
		contents[i][j] = newEntry;
	}
}
