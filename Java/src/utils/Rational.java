package utils;

import java.math.BigInteger;

// Class representing a rational number using arbitrary precision integers.
// Fractions are always stored in normalized (reduced) form with a positive denominator.
public final class Rational {
    private final BigInteger num; // Numerator
    private final BigInteger den; // Denominator (always positive after normalization)
    
    // Main constructor: creates a rational number num/den and normalizes it.
    public Rational(BigInteger num, BigInteger den) {
    	// Prevent division by zero
        if (den.equals(BigInteger.ZERO))
            throw new ArithmeticException("Denominator cannot be zero");

        // Normalize sign so denominator is always positive
        if (den.signum() < 0) {
            num = num.negate();
            den = den.negate();
        }
        
        // Reduce fraction by greatest common divisor
        BigInteger g = num.gcd(den);
        this.num = num.divide(g);
        this.den = den.divide(g);
    }
    
    // Convenience constructor for integer values (n/1)
    public Rational(long n) {
        this(BigInteger.valueOf(n), BigInteger.ONE);
    }
    
    // Returns the sum of this rational and another
    public Rational add(Rational other) {
        return new Rational(
            num.multiply(other.den).add(other.num.multiply(den)),
            den.multiply(other.den)
        );
    }
    
    // Returns the difference between this rational and another
    public Rational subtract(Rational other) {
        return new Rational(
            num.multiply(other.den).subtract(other.num.multiply(den)),
            den.multiply(other.den)
        );
    }
    
    // Returns the product of this rational and another
    public Rational multiply(Rational other) {
        return new Rational(
            num.multiply(other.num),
            den.multiply(other.den)
        );
    }
    
    // Returns the quotient of this rational divided by another
    public Rational divide(Rational other) {
        return new Rational(
            num.multiply(other.den),
            den.multiply(other.num)
        );
    }
    
    // Returns the reciprocal of this rational number
    public Rational reciprocal() {
        return new Rational(den, num);
    }
    
    // Checks whether this rational number is zero
    public boolean isZero() {
        return num.equals(BigInteger.ZERO);
    }
    
    // Returns a convenient string representation
    // If denominator is 1, just show the numerator
    @Override
    public String toString() {
        return den.equals(BigInteger.ONE) ? num.toString() : num + "/" + den;
    }
    
    // Returns a LaTeX formatted representation of the rational number
    // Uses \frac{}{} when denominator is not 1 and handles sign explicitly
    public String toLatexString() {
        if(isPositive()) {
        	return den.equals(BigInteger.ONE) ? num.toString() : "\\frac{" + num + "}{" + den + "}";
        } else {
        	return den.equals(BigInteger.ONE) ? num.toString() : "-\\frac{" + num.negate() + "}{" + den + "}";
        }
    }
    
    // Converts the rational number to a double 
    // CAREFUL: may lose precision
    public double toDouble() {
        return num.doubleValue() / den.doubleValue();
    }
    
    // Raises this rational number to an integer power
    public Rational pow(int pow) {
    	return new Rational(
                num.pow(pow),
                den.pow(pow)
        );
    }
   
    // Returns true if the rational number is strictly positive
    public boolean isPositive() {
    	return num.compareTo(BigInteger.ZERO) > 0;
    }
}

