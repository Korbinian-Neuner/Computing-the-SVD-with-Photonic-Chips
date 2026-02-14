package main;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import utils.MathUtils;
import utils.Matrix;
import utils.Operation;
import utils.Rational;

public class Main {
	
	public static void main(String[] args) {
		//calcPolynomials(15, 5, 8, false);
		
		
		//timeCheckGRK(250, 5, 41);
		//timeCheckQR(250, 5, 41);
		
		//test();
		
	}
	
	public static void timeCheckGRK(int count, int minSize, int maxSize) {
		Logger.setActive(false);
		Random rng = new Random();
		long[][] times = new long[maxSize - minSize][count];
		for(int i = minSize; i < maxSize; i++) {
			System.out.print(i + " ");
			for(int j = 0; j < count; j++) {
				Matrix test = Matrix.createRandomMatrix(i, i, rng.nextLong());
				
				long time = System.currentTimeMillis();
				Matrix[] psvd = SVD.calcSVD(test, (int) 60);
				time = System.currentTimeMillis() - time;
				
				
				times[i - minSize][j] = time;
			}
			System.out.println(i);
		}
		
		for(int i = minSize; i < maxSize; i++) {
			System.out.print(i + " ");
			for(int j = 0; j < count; j++) {
				System.out.print(times[i - minSize][j] + " ");
			}
			System.out.println();
		}
	}
	
	public static void timeCheckQR(int count, int minSize, int maxSize) {
		Logger.setActive(false);
		Random rng = new Random();
		long[][] times = new long[maxSize - minSize][count];
		for(int i = minSize; i < maxSize; i++) {
			System.out.print(i + " ");
			for(int j = 0; j < count; j++) {
				Matrix test = Matrix.createRandomMatrix(i, i, rng.nextLong());
				
				long time = System.currentTimeMillis();
				Matrix[] psvd = DigQRSVD.calcSVD(test, (int) 14f * i);
				time = System.currentTimeMillis() - time;
				
				
				times[i - minSize][j] = time;
			}
			System.out.println(i);
		}
		
		for(int i = minSize; i < maxSize; i++) {
			System.out.print(i + " ");
			for(int j = 0; j < count; j++) {
				System.out.print(times[i - minSize][j] + " ");
			}
			System.out.println();
		}
	}
	
	public static void test() {
		Matrix test = Matrix.createRandomMatrix(15, 8, 19);
		System.out.println(test);
		Matrix[] psvd = PhotonicSVD.calcSVD(test, 1280);
		System.out.println("Diagonal matrix in the middle - should be at least close to diagonal");
		System.out.println(psvd[1]);
		Matrix reconstructed = Matrix.multiply(psvd[0], Matrix.multiply(psvd[1], psvd[2]));
		System.out.println("Reconstructed Matrix");
		System.out.println(reconstructed);
		System.out.println("Difference between original and reconstructed - should be zero");
		System.out.println(Matrix.add(test.scalmul(-1.0), reconstructed));
	}
	
	public static void calcPolynomials(int s1, int s2, int s3, boolean parallelized) {
		
		HashMap<String, int[][][]> counts = new HashMap<String, int[][][]>();
		
		for(int m = s1; m < s1 + 10; m++) {
			for(int n = s2; n < s2 + 10; n++) {
				for(int c = s3; c < s3 + 8; c++) {
					Logger.reset();
					MathUtils.isParallelized = parallelized;
					Matrix test = Matrix.createRandomMatrix(m, n);
					
					// Uncomment Algorithm to count the operations of
					
					//Matrix[] svd = DigQRSVD.calcSVD(test, c); // QR-SVD digital
					//Matrix[] svd = QRSVD.calcSVD(test, c); // QR-SVD hybrid
					Matrix[] svd = SVD.calcSVD(test, c); // GRK-SVD digital
					//Matrix[] svd = PhotonicSVD.calcSVD(test, c); // GRK-SVD hybrid
					
					
					HashMap<ArrayList<Operation>, String> logs;
					if(parallelized) {
						logs = Logger.getParallelizedLogs();
					} else {
						logs = Logger.getLogs();
					}
					 
					
					
					
					Iterator ite = logs.keySet().iterator();
					while(ite.hasNext()) {
						ArrayList<Operation> clog = (ArrayList<Operation>) ite.next();
						if(clog.size() == 0) continue;
						
						HashMap<String, Integer> summedTotals = new HashMap<String, Integer>();
						int maxTimeStep = 0;
						int minTimeStep = Integer.MAX_VALUE;
						Iterator it = clog.iterator();
						while(it.hasNext()) {
							Operation o = (Operation) it.next();
							if(summedTotals.containsKey(o.getType())) {
								summedTotals.put(o.getType(), summedTotals.get(o.getType()) + 1);
							} else {
								summedTotals.put(o.getType(), 1);
							}
							if(o.getTimeStep() > maxTimeStep) {
								maxTimeStep = o.getTimeStep();
							}
							if(o.getTimeStep() < minTimeStep) {
								minTimeStep = o.getTimeStep();
							}
						}
						it = summedTotals.keySet().iterator();
						while(it.hasNext()) {
							String op = (String) it.next();
							String identifier = logs.get(clog);
							identifier += "_" + op;
							
							if(counts.containsKey(identifier)) {
								int[][][] count = counts.get(identifier);
								count[m - s1][n - s2][c-s3] += summedTotals.get(op);
							} else {
								int[][][] count = new int[10][10][8];
								count[m - s1][n - s2][c-s3] += summedTotals.get(op);
								counts.put(identifier, count);
							}
						}
					}
				}
			}
		}
		
		
		Iterator it = counts.keySet().iterator();
		ArrayList<String> latexPols = new ArrayList<String>();
		while(it.hasNext()) {
			String op = (String) it.next();
			int[][][] count = counts.get(op);
			System.out.println(op);
			Rational[][][] coeffs = MathUtils.toPol(count, s1, s2, s3);
			
			String latexPol = "";
			
			for(int i = coeffs[0][0].length - 1; i >= 0; i--) {
				String klammer = "";
				double[][] dc = new double[coeffs.length][coeffs[0].length];
				for(int j = coeffs[0].length - 1; j >= 0; j--) {
					for(int k = coeffs.length - 1; k >= 0; k--) {
						System.out.print(coeffs[j][k][i].toLatexString() + "  ");
						String var = "";
						if(j != 0) var += j == 1 ? "m" : "m^"+j;
						if(k != 0) var += k == 1 ? "n" : "n^"+k;
						String sign = (klammer.isBlank() || !coeffs[j][k][i].isPositive()) ? "" : "+";
						String cof = coeffs[j][k][i].toLatexString();
						if(!var.isBlank() && cof.equals("1")) cof = "";
						if(!var.isBlank() && cof.equals("-1")) cof = "-";
						if(!coeffs[j][k][i].isZero()) klammer += sign + cof + var + " ";
					}
					System.out.println();
				}
				
				if(i != 0 && !klammer.isEmpty()) {
					String c = i == 1 ? "C" : "C^"+i;
					klammer = "(" + klammer + ")" + c;
				}
				String sign = (latexPol.isBlank() || klammer.isEmpty()) ? "" : "+";
				latexPol += sign + klammer;
				
				System.out.println("---------");
			}
			
			latexPol = op + "\n" +latexPol;
			latexPols.add(latexPol);
		}
		
		it = latexPols.iterator();
		while(it.hasNext()) {
			String s = (String) it.next();
			System.out.println(s);
		}
	}
}
